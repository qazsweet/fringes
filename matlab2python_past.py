import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import binary_fill_holes
from scipy.io import savemat
from skimage import morphology, measure
from PIL import Image
import time

# =========================================================================
# Fringe Skeleton Phase Analysis & Surface Reconstruction  (Low-Contrast)
#
# Three-layer strategy for fringes that are too faint for standard detection:
#
#   Layer 1 — CLAHE + DoG + local-std normalisation
#       Boosts local fringe contrast before thresholding, removes the
#       slowly-varying illumination envelope.
#
#   Layer 2 — BW morphological closing + endpoint proximity bridging
#       Before skeletonizing, close small gaps in the binary fringe mask so
#       nearby fragments merge into longer arcs.  After skeletonizing, also
#       bridge any remaining endpoint pairs that are within GAP_BRIDGE_PX of
#       each other (straight-line join), regardless of arc length.
#
#   Layer 3 — Hough Circle fallback
#       If skeleton-based contour detection still yields no closed rings,
#       cv2.HoughCircles() votes for circles directly from the gradient of the
#       normalised image.  Each detected circle is rasterised as a synthetic
#       skeleton ring so the downstream fringe-order / height pipeline runs
#       unchanged.
# =========================================================================

## 0. Parameters
LAMBDA        = 460e-6    # wavelength (mm)
PIXEL_PITCH   = 0.01      # mm / pixel at full sensor resolution
ROI_SLICE     = np.s_[1200:2100, 2000:3100]
DOWNSCALE     = 2          # working-resolution factor (1 = full, 2 = half, 3 = third)
OUTPUT_FRAC   = 0.2        # output grid as fraction of working resolution

# --- Auto-scaled thresholds (keep fringe topology consistent across scales) ---
_sigma      = max(0.5, 2.0 / DOWNSCALE)
_min_obj    = max(8,   50 // (DOWNSCALE * DOWNSCALE))
_margin     = max(3,   15 // DOWNSCALE)
_min_branch = max(3,   10 // DOWNSCALE)

# --- Low-contrast fringe enhancement parameters ---
_clahe_tile  = max(16, 64 // DOWNSCALE)   # CLAHE tile side (px) — ~1 fringe period
_clahe_clip  = 3.0                         # CLAHE contrast clip limit
_dog_sigma1  = max(0.8, 1.5 / DOWNSCALE)  # DoG fine sigma (noise removal)
_dog_sigma2  = max(20,  80 // DOWNSCALE)   # DoG coarse sigma (background removal)

# --- Radial profile ring detection ---
# Minimum pixel distance between adjacent rings in the radial profile.
# Set to ~half the minimum expected fringe spacing (in working-resolution px).
_ring_min_dist   = max(3, 8 // DOWNSCALE)
# Minimum prominence of a profile minimum to count as a fringe ring.
# Lower = more sensitive (more rings), but also more noise.  Increase if
# spurious rings appear in flat/noisy regions.
_ring_prominence = 0.1
# Minimum fraction of a circle's arc that must lie inside the image for
# that radius to be included in the radial profile.  Allows detection of
# rings whose centre is near or outside the ROI edge.
_min_arc_frac    = 0.20   # 20 % — rejects tiny slivers with unreliable mean
# Pixels within this many pixels of the image boundary are excluded from
# the radial profile to avoid CLAHE / DoG normalisation edge artefacts.
_profile_margin  = 5


def _lap(label, t_prev):
    t = time.time()
    print(f"  {label:.<30s} {t - t_prev:.3f}s")
    return t

def process_image(img_name, img_file):
    # ── 1. Load & resize ─────────────────────────────────────────────────────
    t0 = time.time()

    img_roi = np.array(Image.open(img_name).convert('L'))[ROI_SLICE]
    if DOWNSCALE > 1:
        img_roi = cv2.resize(
            img_roi,
            (img_roi.shape[1] // DOWNSCALE, img_roi.shape[0] // DOWNSCALE),
            interpolation=cv2.INTER_AREA,
        )
    img_u8 = img_roi  # keep uint8 for CLAHE
    img_f = img_roi.astype(np.float32) / 255.0
    H, W = img_f.shape
    t = _lap("Load & resize", t0)
    print(f"  Working size: {W} × {H}")

    # ── 2. CLAHE — locally boost fringe contrast in bright/dim regions ────────
    # Works on the 8-bit image; tile size is set so each tile covers roughly
    # one fringe period, giving per-fringe contrast normalisation.
    clahe = cv2.createCLAHE(clipLimit=_clahe_clip,
                             tileGridSize=(_clahe_tile, _clahe_tile))
    img_clahe = clahe.apply(img_u8).astype(np.float32) / 255.0
    t = _lap("CLAHE", t)

    # ── 3. DoG background removal + local-std normalisation ──────────────────
    # DoG = blur_fine − blur_coarse  isolates the fringe AC signal by
    # subtracting the slowly-varying illumination envelope.  Dividing by the
    # local standard deviation then equalises sensitivity across regions.
    ksize1 = int(np.ceil(_dog_sigma1 * 6)) | 1
    ksize2 = int(np.ceil(_dog_sigma2 * 6)) | 1
    blur_fine   = cv2.GaussianBlur(img_clahe, (ksize1, ksize1), _dog_sigma1)
    blur_coarse = cv2.GaussianBlur(img_clahe, (ksize2, ksize2), _dog_sigma2)
    dog = blur_fine - blur_coarse   # fringe signal, DC removed

    # Local standard deviation — computed as sqrt(E[x²] − E[x]²) with a
    # Gaussian window of width _dog_sigma2 (same scale as background).
    local_mean = cv2.GaussianBlur(dog, (ksize2, ksize2), _dog_sigma2)
    local_sq   = cv2.GaussianBlur(dog * dog, (ksize2, ksize2), _dog_sigma2)
    local_std  = np.sqrt(np.maximum(local_sq - local_mean * local_mean, 0) + 1e-6)
    img_norm   = (dog - local_mean) / local_std   # zero-mean, unit-variance
    t = _lap("DoG + local-std normalisation", t)

    # ── 4. Ring centre detection via gradient-direction consistency ───────────
    # At the true centre of a circular fringe pattern the image gradient at
    # every edge pixel points radially (toward or away from the centre).
    # We vote for the centre on a coarse grid by measuring how well each
    # candidate aligns with the gradient directions of strong-edge pixels.
    gx = cv2.Sobel(img_norm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_norm, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy) + 1e-6

    # Use only the top 15 % strongest gradient pixels (fringe edges)
    thresh = np.percentile(grad_mag, 85)
    ep_ys, ep_xs = np.where(grad_mag > thresh)
    ep_xs_f = ep_xs.astype(np.float64)
    ep_ys_f = ep_ys.astype(np.float64)
    gx_n = (gx[ep_ys, ep_xs] / grad_mag[ep_ys, ep_xs]).astype(np.float64)
    gy_n = (gy[ep_ys, ep_xs] / grad_mag[ep_ys, ep_xs]).astype(np.float64)

    search_step = max(5, 10 // DOWNSCALE)
    best_score = -np.inf
    best_cx, best_cy = W // 2, H // 2

    for cy_c in range(H // 4, 3 * H // 4, search_step):
        for cx_c in range(W // 4, 3 * W // 4, search_step):
            dx = float(cx_c) - ep_xs_f
            dy = float(cy_c) - ep_ys_f
            dist = np.sqrt(dx * dx + dy * dy) + 1e-6
            # |cos θ| = alignment between gradient and radial direction
            score = float(np.abs(gx_n * (dx / dist) + gy_n * (dy / dist)).mean())
            if score > best_score:
                best_score = score
                best_cx, best_cy = cx_c, cy_c

    cx, cy = best_cx, best_cy
    print(f"  Ring centre: ({cx}, {cy})")
    t = _lap("Centre detection", t)

    # ── 5. Radial profile → ring radii ───────────────────────────────────────
    # Compute the mean normalised value on concentric circles at each radius.
    # Dark fringes (destructive interference) appear as local minima.
    #
    # When the ring centre is near the image edge (as happens when the
    # interference pattern shifts during a sequence), many rings extend
    # partially outside the ROI.  We allow radii up to the furthest corner and
    # sample only the in-bounds arc fraction, discarding radii where fewer than
    # _min_arc_frac of the circle is inside the image.
    from scipy.signal import find_peaks

    # Maximum radius: furthest corner from centre (full arc or partial arc)
    max_r = int(np.ceil(max(
        np.hypot(cx,     cy),
        np.hypot(W - cx, cy),
        np.hypot(cx,     H - cy),
        np.hypot(W - cx, H - cy),
    )))

    r_vals  = np.arange(1, max_r + 1)
    profile = np.full(len(r_vals), np.nan, dtype=np.float64)

    pm = _profile_margin
    for i, r in enumerate(r_vals):
        n_ang = max(32, int(2 * np.pi * r))
        ang   = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
        xs    = np.round(cx + r * np.cos(ang)).astype(int)
        ys    = np.round(cy + r * np.sin(ang)).astype(int)
        # Exclude pixels within _profile_margin of the image boundary to
        # avoid CLAHE / DoG normalisation edge artefacts.
        valid = (xs >= pm) & (xs < W - pm) & (ys >= pm) & (ys < H - pm)
        # Require at least _min_arc_frac of the circle to be inside the image
        if valid.sum() >= _min_arc_frac * n_ang:
            profile[i] = img_norm[ys[valid], xs[valid]].mean()

    # Interpolate NaN gaps (arcs mostly outside the image) so find_peaks
    # doesn't stumble on discontinuities
    nan_mask = np.isnan(profile)
    if nan_mask.any() and (~nan_mask).sum() >= 2:
        profile[nan_mask] = np.interp(
            r_vals[nan_mask], r_vals[~nan_mask], profile[~nan_mask]
        )

    # Find minima of profile (dark fringes).  Tune prominence/distance if
    # too many or too few rings are detected.
    ring_idxs, _ = find_peaks(-profile,
                               distance=_ring_min_dist,
                               prominence=_ring_prominence)
    ring_radii = r_vals[ring_idxs]
    print(f"  Rings found: {len(ring_radii)}  radii={ring_radii.tolist()}")

    if len(ring_radii) == 0:
        raise ValueError("Radial profile found no fringe rings.  "
                         "Try lowering _ring_prominence or _ring_min_dist.")
    t = _lap("Radial profile", t)

    # ── 6. Extract real contours within each ring's annular band ─────────────
    # The radial profile tells us *where* each ring is; we now go back to the
    # normalised image and extract the actual dark-fringe pixels inside an
    # annular mask around each detected radius.  This gives real contours that
    # follow the true fringe shape (not perfect circles).
    #
    # For each ring i:
    #   band = annular mask from r_lo to r_hi  (half-spacing to neighbours)
    #   within that band, threshold img_norm < 0  (dark fringe)
    #   skeletonize → 1-px contour
    #   assign fringe order = i+1

    # Pre-compute distance-from-centre for every pixel once
    yy, xx = np.mgrid[0:H, 0:W]
    dist_map = np.hypot(xx - cx, yy - cy).astype(np.float32)

    # Fringe validity mask: only extract contours where there is actual
    # illumination.  Pixels where the smoothed original image is darker than
    # 20 % of the ROI's max brightness carry no useful fringe information —
    # they are outside the beam footprint.  This suppresses spurious contours
    # in dark / unilluminated corners.
    illum = cv2.GaussianBlur(img_f, (0, 0), float(_dog_sigma2))
    valid_illum = (illum > 0.20 * illum.max()).astype(np.uint8)

    skel_all  = np.zeros((H, W), dtype=bool)
    order_map = np.zeros((H, W), dtype=int)   # fringe order at each pixel

    for i, r in enumerate(ring_radii):
        r_lo = (ring_radii[i - 1] + r) / 2.0 if i > 0               else max(0, r - (ring_radii[1] - ring_radii[0]) / 2.0 if len(ring_radii) > 1 else r * 0.5)
        r_hi = (r + ring_radii[i + 1]) / 2.0 if i < len(ring_radii) - 1 else r + (r - ring_radii[i - 1]) / 2.0 if i > 0 else r * 1.5

        band = (dist_map >= r_lo) & (dist_map <= r_hi)

        # Dark fringe pixels within this band, restricted to illuminated area
        fringe_bw = ((img_norm < 0) & band & valid_illum.astype(bool)).astype(np.uint8)

        # Remove tiny noise blobs
        n_cc, cc_map, stats, _ = cv2.connectedComponentsWithStats(fringe_bw, 8)
        small = np.where(stats[1:, cv2.CC_STAT_AREA] < _min_obj)[0] + 1
        if small.size:
            fringe_bw[np.isin(cc_map, small)] = 0

        if fringe_bw.sum() == 0:
            continue

        # Skeletonize to get 1-pixel wide contour
        skel_i = morphology.skeletonize(fringe_bw.astype(bool))

        skel_all |= skel_i
        order_map[skel_i] = i + 1

    if skel_all.sum() == 0:
        raise ValueError("No real fringe contour pixels extracted from any ring band.")

    row_skel, col_skel = np.where(skel_all)
    bin_idx  = order_map[row_skel, col_skel]
    x_center, y_center = float(cx), float(cy)
    print(f"  Extremum at: ({x_center:.2f}, {y_center:.2f})")
    print(f"  Real contour pixels: {skel_all.sum()}")

    z_skel = bin_idx * (LAMBDA / 2.0)
    t = _lap("Real contour extraction", t)

    plt.figure("Real fringe contours")
    plt.imshow(img_roi, cmap='gray')
    plt.imshow(skel_all, cmap='autumn', alpha=0.7)
    plt.scatter([cx], [cy], c='r', s=60, marker='+')
    plt.title(f"Real contours ({len(ring_radii)} rings, {skel_all.sum()} px)")
    plt.show(block=False)

    # ── 8. Surface interpolation on reduced output grid ──────────────────────
    phys_px = PIXEL_PITCH * DOWNSCALE
    points = np.column_stack((col_skel * phys_px, row_skel * phys_px))

    nx = max(10, int(W * OUTPUT_FRAC))
    ny = max(10, int(H * OUTPUT_FRAC))
    gx, gy = np.meshgrid(
        np.linspace(0, W * phys_px, nx),
        np.linspace(0, H * phys_px, ny),
    )

    nz = z_skel != 0
    z_surf = griddata(points[nz], z_skel[nz], (gx, gy), method='linear')
    t = _lap("Interpolation", t)

    print(f"\n  Output shape : {z_surf.shape}")
    print(f"  Total time   : {t - t0:.3f}s")

    savemat(f'data_{img_file.split(".")[0]}.mat', {
        'x_center': x_center,
        'y_center': y_center,
        'z_surface': z_surf,
    })

    from mpl_toolkits.mplot3d import Axes3D
    # 3D Surface Plot
    fig = plt.figure(f"3D Surface {img_file.split(".")[0]}")
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(gx, gy, z_surf, cmap='viridis', edgecolor='none')
    ax.set_title("Reconstructed Surface (3D)")
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    plt.show()



def main():

    import os
    image_folder =  './20260330_215004' #'./20260330_215619' # 
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith('.png')])

    if image_files:
        print(f'found {len(image_files)} images in {image_folder}')

    else:
        print(f'no images found in {image_folder}')
        return

    for img_file in image_files:
        img_name = os.path.join(image_folder, img_file)
        print(f"Processing {img_name} ===========================================")
        try:
            process_image(img_name, img_file)
        except Exception as e:
            print(f"  Error processing {img_file}: {e}")
            print("  Skipping this file.\n")
            continue
   


if __name__ == '__main__':

    main()