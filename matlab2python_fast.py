import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.io import savemat
from scipy.signal import find_peaks
from skimage import morphology
from PIL import Image
import time

# =========================================================================
# Fringe Skeleton Phase Analysis & Surface Reconstruction  (Low-Contrast)
# Optimised version of matlab2python_past.py:
#
#   Step 3  DoG blurs: keep cv2.GaussianBlur (SIMD wins at 247k px)
#   Step 4  Centre: top-3% edge pixels + coarse-to-fine grid search
#   Step 5  Radial profile: np.bincount on dist_map (no per-radius loop)
#   Step 6  illum mask: computed at 1/4 resolution (16× fewer pixels)
#   Step 6  Skeletonise: cv2.ximgproc.thinning (faster C++), single call
# =========================================================================

## 0. Parameters
LAMBDA        = 460e-6    # wavelength (mm)
PIXEL_PITCH   = 0.01      # mm / pixel at full sensor resolution
ROI_SLICE     = np.s_[1200:2100, 2000:3100]
DOWNSCALE     = 2
OUTPUT_FRAC   = 0.2

_sigma      = max(0.5, 2.0 / DOWNSCALE)
_min_obj    = max(8,   50 // (DOWNSCALE * DOWNSCALE))
_margin     = max(3,   15 // DOWNSCALE)
_min_branch = max(3,   10 // DOWNSCALE)

_clahe_tile  = max(16, 64 // DOWNSCALE)
_clahe_clip  = 3.0
_dog_sigma1  = max(0.8, 1.5 / DOWNSCALE)
_dog_sigma2  = max(20,  80 // DOWNSCALE)

_ring_min_dist   = max(3, 8 // DOWNSCALE)
_ring_prominence = 0.1
_min_arc_frac    = 0.20
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
    img_u8 = img_roi
    img_f  = img_roi.astype(np.float32) / 255.0
    H, W   = img_f.shape
    t = _lap("Load & resize", t0)
    print(f"  Working size: {W} × {H}")

    # ── 2. CLAHE ─────────────────────────────────────────────────────────────
    clahe     = cv2.createCLAHE(clipLimit=_clahe_clip,
                                 tileGridSize=(_clahe_tile, _clahe_tile))
    img_clahe = clahe.apply(img_u8).astype(np.float32) / 255.0
    t = _lap("CLAHE", t)

    # ── 3. DoG + local-std normalisation ─────────────────────────────────────
    # cv2.GaussianBlur uses SIMD-optimised separable convolution; at 247k pixels
    # it outperforms iterated box filters despite the large kernel (k≈241).
    ksize1 = int(np.ceil(_dog_sigma1 * 6)) | 1
    ksize2 = int(np.ceil(_dog_sigma2 * 6)) | 1
    blur_fine   = cv2.GaussianBlur(img_clahe, (ksize1, ksize1), _dog_sigma1)
    blur_coarse = cv2.GaussianBlur(img_clahe, (ksize2, ksize2), _dog_sigma2)
    dog         = blur_fine - blur_coarse

    local_mean = cv2.GaussianBlur(dog,       (ksize2, ksize2), _dog_sigma2)
    local_sq   = cv2.GaussianBlur(dog * dog, (ksize2, ksize2), _dog_sigma2)
    local_std  = np.sqrt(np.maximum(local_sq - local_mean * local_mean, 0) + 1e-6)
    img_norm   = (dog - local_mean) / local_std
    t = _lap("DoG + local-std", t)

    # ── 4. Ring centre detection (OPTIMISED) ─────────────────────────────────
    # Two changes that reduce work without changing the algorithm:
    #   a) Use top 3% gradient pixels instead of top 15% — the alignment score
    #      is a mean and degrades gracefully; 5× fewer points per iteration.
    #   b) Coarse-to-fine search: pass 1 at large step finds a rough centre;
    #      pass 2 refines in a tight window at fine step.  ~5× fewer total iters.
    gx_s = cv2.Sobel(img_norm, cv2.CV_32F, 1, 0, ksize=3)
    gy_s = cv2.Sobel(img_norm, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx_s * gx_s + gy_s * gy_s) + 1e-6

    thresh   = np.percentile(grad_mag, 97)   # top 3%
    ep_ys, ep_xs = np.where(grad_mag > thresh)
    ep_xs_f  = ep_xs.astype(np.float64)
    ep_ys_f  = ep_ys.astype(np.float64)
    gx_n     = (gx_s[ep_ys, ep_xs] / grad_mag[ep_ys, ep_xs]).astype(np.float64)
    gy_n     = (gy_s[ep_ys, ep_xs] / grad_mag[ep_ys, ep_xs]).astype(np.float64)

    def _score(cx_c, cy_c):
        dx   = float(cx_c) - ep_xs_f
        dy   = float(cy_c) - ep_ys_f
        dist = np.sqrt(dx * dx + dy * dy) + 1e-6
        return float(np.abs(gx_n * (dx / dist) + gy_n * (dy / dist)).mean())

    # Pass 1 — coarse grid
    _COARSE = max(15, 30 // DOWNSCALE)
    best_score = -np.inf
    best_cx, best_cy = W // 2, H // 2
    for cy_c in range(H // 4, 3 * H // 4, _COARSE):
        for cx_c in range(W // 4, 3 * W // 4, _COARSE):
            s = _score(cx_c, cy_c)
            if s > best_score:
                best_score = s
                best_cx, best_cy = cx_c, cy_c

    # Pass 2 — fine grid within ±_COARSE of coarse best
    _FINE = max(2, 4 // DOWNSCALE)
    for cy_c in range(best_cy - _COARSE, best_cy + _COARSE + 1, _FINE):
        for cx_c in range(best_cx - _COARSE, best_cx + _COARSE + 1, _FINE):
            if 0 <= cx_c < W and 0 <= cy_c < H:
                s = _score(cx_c, cy_c)
                if s > best_score:
                    best_score = s
                    best_cx, best_cy = cx_c, cy_c

    cx, cy = best_cx, best_cy
    print(f"  Ring centre: ({cx}, {cy})")
    t = _lap("Centre detection (coarse-to-fine)", t)

    # ── 5. Radial profile → ring radii (OPTIMISED) ───────────────────────────
    # Replacement: precompute dist_map once (reused in step 6), then use
    # np.bincount for O(N) accumulation instead of a per-radius Python loop.
    # Arc-fraction check is preserved: require that at least _min_arc_frac of
    # the pixels at that discrete radius lie inside the valid margin region.
    yy, xx = np.mgrid[0:H, 0:W]

    max_r    = int(np.ceil(max(
        np.hypot(cx,     cy),
        np.hypot(W - cx, cy),
        np.hypot(cx,     H - cy),
        np.hypot(W - cx, H - cy),
    )))

    dist_map = np.hypot(xx - cx, yy - cy).astype(np.float32)
    r_idx    = np.round(dist_map).astype(np.int32).ravel()   # integer radius per px
    r_idx_c  = np.clip(r_idx, 0, max_r)

    pm          = _profile_margin
    in_bounds   = (
        (xx.ravel() >= pm) & (xx.ravel() < W - pm) &
        (yy.ravel() >= pm) & (yy.ravel() < H - pm)
    )

    vals_flat   = img_norm.ravel()

    # Sum of img_norm and pixel count for in-bounds pixels only
    sum_ib  = np.bincount(r_idx_c[in_bounds],
                          weights=vals_flat[in_bounds],
                          minlength=max_r + 1)
    cnt_ib  = np.bincount(r_idx_c[in_bounds], minlength=max_r + 1).astype(np.float64)
    # Total pixels at each integer radius (full image grid) for arc-frac denominator
    cnt_all = np.bincount(r_idx_c,            minlength=max_r + 1).astype(np.float64)

    r_vals  = np.arange(1, max_r + 1)
    arc_frac = cnt_ib[r_vals] / np.maximum(cnt_all[r_vals], 1)
    profile  = np.where(
        arc_frac >= _min_arc_frac,
        sum_ib[r_vals] / np.maximum(cnt_ib[r_vals], 1),
        np.nan,
    )

    # Interpolate NaN gaps
    nan_mask = np.isnan(profile)
    if nan_mask.any() and (~nan_mask).sum() >= 2:
        profile[nan_mask] = np.interp(
            r_vals[nan_mask], r_vals[~nan_mask], profile[~nan_mask]
        )

    ring_idxs, _ = find_peaks(-profile,
                               distance=_ring_min_dist,
                               prominence=_ring_prominence)
    ring_radii = r_vals[ring_idxs]
    print(f"  Rings found: {len(ring_radii)}  radii={ring_radii.tolist()}")

    if len(ring_radii) == 0:
        raise ValueError("Radial profile found no fringe rings. "
                         "Try lowering _ring_prominence or _ring_min_dist.")
    t = _lap("Radial profile (bincount)", t)

    # ── 6. Real contour extraction — single skeletonise (OPTIMISED) ──────────
    # illum validity mask: illumination is slowly varying, so computing the
    # large-sigma blur at 1/4 resolution (16× fewer pixels) is equivalent.
    Hs, Ws      = max(1, H // 4), max(1, W // 4)
    illum_small = cv2.GaussianBlur(
        cv2.resize(img_f, (Ws, Hs), interpolation=cv2.INTER_AREA),
        (0, 0), _dog_sigma2 / 4.0,
    )
    illum       = cv2.resize(illum_small, (W, H), interpolation=cv2.INTER_LINEAR)
    valid_illum = illum > 0.20 * illum.max()

    all_fringe_bw   = np.zeros((H, W), dtype=np.uint8)
    band_assignment = np.zeros((H, W), dtype=np.int32)

    for i, r in enumerate(ring_radii):
        r_lo = (
            (ring_radii[i - 1] + r) / 2.0
            if i > 0
            else max(0.0, r - (ring_radii[1] - ring_radii[0]) / 2.0
                     if len(ring_radii) > 1 else r * 0.5)
        )
        r_hi = (
            (r + ring_radii[i + 1]) / 2.0
            if i < len(ring_radii) - 1
            else r + (r - ring_radii[i - 1]) / 2.0
            if i > 0 else r * 1.5
        )

        band      = (dist_map >= r_lo) & (dist_map <= r_hi)
        fringe_bw = ((img_norm < 0) & band & valid_illum).astype(np.uint8)

        n_cc, cc_map, stats, _ = cv2.connectedComponentsWithStats(fringe_bw, 8)
        small = np.where(stats[1:, cv2.CC_STAT_AREA] < _min_obj)[0] + 1
        if small.size:
            fringe_bw[np.isin(cc_map, small)] = 0

        mask = fringe_bw.astype(bool)
        all_fringe_bw  |= fringe_bw
        band_assignment[mask] = i + 1   # fringe order stored before skeletonise

    if all_fringe_bw.sum() == 0:
        raise ValueError("No real fringe contour pixels extracted from any ring band.")

    # Single thinning call — cv2.ximgproc.thinning uses Guo-Hall in optimised
    # C++, typically 3-5× faster than skimage.skeletonize on the same mask.
    # Falls back to skimage if opencv-contrib is not installed.
    try:
        skel_all = cv2.ximgproc.thinning(all_fringe_bw * 255).astype(bool)
    except AttributeError:
        skel_all = morphology.skeletonize(all_fringe_bw.astype(bool))
    order_map = np.where(skel_all, band_assignment, 0)

    row_skel, col_skel = np.where(skel_all)
    bin_idx    = order_map[row_skel, col_skel]
    x_center, y_center = float(cx), float(cy)
    print(f"  Extremum at: ({x_center:.2f}, {y_center:.2f})")
    print(f"  Real contour pixels: {skel_all.sum()}")

    z_skel = bin_idx * (LAMBDA / 2.0)
    t = _lap("Real contour (1 skeletonise)", t)

    plt.figure("Real fringe contours")
    plt.imshow(img_roi, cmap='gray')
    plt.imshow(skel_all, cmap='autumn', alpha=0.7)
    plt.scatter([cx], [cy], c='r', s=60, marker='+')
    plt.title(f"Real contours ({len(ring_radii)} rings, {skel_all.sum()} px)")
    plt.show(block=False)

    # ── 7. Surface interpolation ─────────────────────────────────────────────
    phys_px = PIXEL_PITCH * DOWNSCALE
    points  = np.column_stack((col_skel * phys_px, row_skel * phys_px))

    nx = max(10, int(W * OUTPUT_FRAC))
    ny = max(10, int(H * OUTPUT_FRAC))
    gx_g, gy_g = np.meshgrid(
        np.linspace(0, W * phys_px, nx),
        np.linspace(0, H * phys_px, ny),
    )

    nz     = z_skel != 0
    z_surf = griddata(points[nz], z_skel[nz], (gx_g, gy_g), method='linear')
    t = _lap("Interpolation", t)

    print(f"\n  Output shape : {z_surf.shape}")
    print(f"  Total time   : {t - t0:.3f}s")

    savemat(f'data_{img_file.split(".")[0]}.mat', {
        'x_center':  x_center,
        'y_center':  y_center,
        'z_surface': z_surf,
    })

    fig = plt.figure(f"3D Surface {img_file.split('.')[0]}")
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot_surface(gx_g, gy_g, z_surf, cmap='viridis', edgecolor='none')
    ax.set_title("Reconstructed Surface (3D)")
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    plt.show()


def main():
    import os
    image_folder = './20260330_215004'
    image_files  = sorted([f for f in os.listdir(image_folder)
                            if f.lower().endswith('.png')])

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
