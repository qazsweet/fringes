import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import time

# =========================================================================
# FFT-based fringe onset detector
#
# For each flat-field-corrected Field frame, compute the 2-D FFT of the ROI
# and measure the total power in the spatial-frequency band that corresponds
# to interference fringe periods.  Track this "fringe energy" metric across
# all frames, establish a quiet baseline from the leading frames, and report
# the first frame where energy exceeds threshold = baseline + k * baseline_std.
#
# Outputs
#   Output/fft_energy_plot.png   – fringe energy vs frame (log scale)
#   Output/fft_2d_<frame>.png    – 2-D FFT magnitude for key frames
# =========================================================================

_SCIMAGE   = '/Users/erinsweet/Downloads/codes/align/fringes/SCImage'
BRIGHT_DIR = os.path.join(_SCIMAGE, 'Bright')
DARK_DIR   = os.path.join(_SCIMAGE, 'Dark')
FIELD_DIR  = os.path.join(_SCIMAGE, 'Field')
OUT_DIR    = os.path.join(_SCIMAGE, 'Output')

ROI_SLICE  = np.s_[1200:2100, 2000:3100]   # (900 × 1100) region of interest

# Fringe spatial-frequency band (in cycles per ROI height H)
# Fringe periods: 30–300 px in the full-res ROI → freq = H / period
FRINGE_PERIOD_MIN_PX = 30    # highest spatial frequency (finest fringes)
FRINGE_PERIOD_MAX_PX = 300   # lowest spatial frequency  (coarsest fringes)

# Onset detection: first frame whose energy exceeds baseline + K * baseline_std
BASELINE_N_FRAMES = 5        # number of leading quiet frames used for baseline
ONSET_K           = 6.0      # sigma multiplier for onset threshold


def _ts():
    t = time.time()
    return time.strftime('%H:%M:%S', time.localtime(t)) + f'.{int((t % 1)*1000):03d}'


def load_mean(folder):
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.bmp')])
    if not files:
        raise FileNotFoundError(f"No BMP files in {folder}")
    stack = [np.array(Image.open(os.path.join(folder, f))).astype(np.float32)
             for f in files]
    return np.mean(stack, axis=0)


def flat_field_correct(raw_f32, mean_dark, divisor):
    corr = (raw_f32 - mean_dark) / divisor
    return np.clip(corr, 0, None)


def fringe_band_energy(roi_f32):
    """
    Compute total PSD power in the fringe spatial-frequency band.

    Steps:
      1. Apply 2-D Hann window to suppress spectral leakage at ROI edges.
      2. 2-D FFT → shift DC to centre → |FFT|^2.
      3. Radially average the PSD (concentric-ring fringes have isotropic
         frequency content, so the radial average is the natural summary).
      4. Sum the radial bins whose period falls in [FRINGE_PERIOD_MIN_PX,
         FRINGE_PERIOD_MAX_PX].

    Returns
      energy   : float  – total fringe-band power
      psd_2d   : 2-D array  – |FFT|^2 (shifted), for visualisation
      radial   : 1-D array  – radially averaged PSD
      r_vals   : 1-D array  – corresponding radius / freq index
    """
    H, W = roi_f32.shape

    # 2-D Hann window
    win  = np.outer(np.hanning(H), np.hanning(W)).astype(np.float32)
    spec = np.fft.fft2(roi_f32 * win)
    psd  = np.abs(np.fft.fftshift(spec)) ** 2

    # Radial average
    cy, cx = H // 2, W // 2
    yy, xx = np.mgrid[0:H, 0:W]
    rr     = np.round(np.hypot(xx - cx, yy - cy)).astype(np.int32)
    max_r  = min(cx, cy)

    rr_clip = np.clip(rr.ravel(), 0, max_r)
    psd_sum = np.bincount(rr_clip, weights=psd.ravel(), minlength=max_r + 1)
    psd_cnt = np.bincount(rr_clip, minlength=max_r + 1).astype(np.float64)
    psd_cnt = np.where(psd_cnt == 0, 1, psd_cnt)
    radial  = psd_sum[:max_r] / psd_cnt[:max_r]
    r_vals  = np.arange(len(radial))

    # Frequency band: radius = H / period_px
    f_lo = max(1, int(H / FRINGE_PERIOD_MAX_PX))
    f_hi = min(len(radial) - 1, int(H / FRINGE_PERIOD_MIN_PX) + 1)

    energy = float(radial[f_lo:f_hi].sum())
    return energy, psd, radial, r_vals, f_lo, f_hi


def save_2d_fft(psd_2d, frame_name, label, out_dir):
    """Save log-scale 2-D FFT magnitude image with fringe-band annotation."""
    log_psd = np.log1p(psd_2d)
    log_psd = (log_psd / log_psd.max() * 255).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(log_psd, cmap='inferno', origin='upper')
    H, W = psd_2d.shape
    cy, cx = H // 2, W // 2
    f_lo = max(1, int(H / FRINGE_PERIOD_MAX_PX))
    f_hi = min(H // 2 - 1, int(H / FRINGE_PERIOD_MIN_PX) + 1)
    for r, ls in [(f_lo, '--'), (f_hi, '-')]:
        theta = np.linspace(0, 2 * np.pi, 360)
        ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta),
                color='cyan', lw=1, ls=ls, alpha=0.8)
    ax.set_title(f"2-D FFT  {label}\n(cyan circles = fringe band bounds)", fontsize=9)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'fft_2d_{frame_name}.png'), dpi=120,
                bbox_inches='tight')
    plt.close(fig)


def main():
    t_total = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Calibration ──────────────────────────────────────────────────────────
    print(f"[{_ts()}] Loading calibration frames ...")
    mean_dark   = load_mean(DARK_DIR)
    mean_bright = load_mean(BRIGHT_DIR)
    divisor     = mean_bright - mean_dark
    divisor     = np.where(divisor < 1.0, 1.0, divisor)
    divisor    /= divisor.mean()
    print(f"[{_ts()}] Calibration ready.")

    field_files = sorted([f for f in os.listdir(FIELD_DIR)
                          if f.lower().endswith('.bmp')])
    n = len(field_files)
    print(f"[{_ts()}] Found {n} field frames.\n")

    print(f"  {'#':>3}  {'timestamp':>16}  {'energy':>18}  {'rel_baseline':>13}  {'fringes?':>8}  elapsed")
    print(f"  {'-'*3}  {'-'*16}  {'-'*18}  {'-'*13}  {'-'*8}  -------")

    # ── Per-frame streaming state ─────────────────────────────────────────────
    baseline_buf = []          # energies of first BASELINE_N_FRAMES frames
    base_mean    = None        # locked once baseline window is full
    base_std     = None
    threshold    = None
    energies     = []          # all energies (for final plot only)
    onset_idx    = None        # first YES frame

    # PSD buffer: keep only the frames needed for the final plot.
    # We always want: frame 0, the frame before onset, the onset frame,
    # the frame after onset, and the last frame.  Since we don't know onset
    # in advance, keep a 2-frame sliding window plus frame 0.
    psd_store    = {}          # {frame_idx: (psd_2d, radial, r_vals, f_lo, f_hi)}
    prev_psd     = None        # (i-1) entry, promoted to psd_store on YES
    r_vals_ref   = None
    f_lo_ref     = f_hi_ref = None

    for i, fname in enumerate(field_files):
        t0 = time.time()

        # Load and flat-field correct this frame only
        raw  = np.array(Image.open(os.path.join(FIELD_DIR, fname))).astype(np.float32)
        corr = flat_field_correct(raw, mean_dark, divisor)
        roi  = corr[ROI_SLICE]

        energy, psd_2d, radial, r_vals, f_lo, f_hi = fringe_band_energy(roi)
        energies.append(energy)

        if r_vals_ref is None:
            r_vals_ref = r_vals
            f_lo_ref, f_hi_ref = f_lo, f_hi

        # ── Baseline accumulation ─────────────────────────────────────────────
        if i < BASELINE_N_FRAMES:
            baseline_buf.append(energy)
            flag = 'NO'
            rel  = float('nan')

            # Lock baseline the moment the window is full
            if len(baseline_buf) == BASELINE_N_FRAMES:
                base_mean = float(np.mean(baseline_buf))
                base_std  = float(np.std(baseline_buf))
                threshold = base_mean + ONSET_K * base_std
                print(f"\n[{_ts()}] Baseline locked after frame {i}:  "
                      f"mean={base_mean:.3e}  std={base_std:.3e}  "
                      f"threshold={threshold:.3e}\n")

        # ── Online classification (no future frames used) ─────────────────────
        else:
            rel  = energy / base_mean
            flag = 'YES' if energy > threshold else 'NO'

            if flag == 'YES' and onset_idx is None:
                onset_idx = i
                onset_ts  = fname.split('_')[-1].replace('.bmp', '')
                print(f"\n[{_ts()}] *** Fringe onset detected at frame {i} ***")
                print(f"           Filename  : {fname}")
                print(f"           Timestamp : {onset_ts}")
                print(f"           Energy    : {energy:.3e}  ({rel:.1f}× baseline)\n")

        # ── PSD bookkeeping for plots ─────────────────────────────────────────
        entry = (psd_2d, radial, r_vals, f_lo, f_hi)
        if i == 0:
            psd_store[0] = entry                   # always keep frame 0
        if onset_idx is not None:
            # keep onset-1 (already in prev_psd), onset, onset+1
            if i == onset_idx:
                if prev_psd is not None:
                    psd_store[i - 1] = prev_psd
                psd_store[i] = entry
            elif i == onset_idx + 1:
                psd_store[i] = entry
        prev_psd = entry                            # slide the window

        elapsed  = time.time() - t0
        ts_label = fname.split('_')[-1].replace('.bmp', '')
        rel_str  = f'{rel:13.2f}' if not np.isnan(rel) else f'{"(baseline)":>13}'
        print(f"[{_ts()}]  {i:>3}  {ts_label:>16}  {energy:18.1f}  {rel_str}  {flag:>8}  {elapsed:.3f}s")

    # Keep last frame PSD for the plot
    psd_store[n - 1] = prev_psd

    energies = np.array(energies, dtype=np.float64)

    if onset_idx is None:
        print(f"\n[{_ts()}] No onset detected above threshold in {n} frames.")

    # ── 2-D FFT images for stored key frames ─────────────────────────────────
    for idx in sorted(psd_store):
        psd_2d, *_ = psd_store[idx]
        ts_label   = field_files[idx].split('_')[-1].replace('.bmp', '')
        label      = f"frame {idx}  ts={ts_label}"
        if idx == onset_idx:
            label += '  ← ONSET'
        print(f"[{_ts()}] Saving 2-D FFT for {label} ...")
        save_2d_fft(psd_2d, f'{idx:02d}_{ts_label}', label, OUT_DIR)

    # ── Energy vs frame plot ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    ax = axes[0]
    ax.semilogy(range(n), energies, 'o-', color='steelblue', lw=1.5,
                markersize=4, label='Fringe-band energy')
    ax.axhline(threshold, color='red', lw=1.2, ls='--',
               label=f'Threshold (baseline + {ONSET_K}σ)')
    ax.axhspan(0, threshold, alpha=0.06, color='green', label='Baseline region')
    if onset_idx is not None:
        ax.axvline(onset_idx, color='orange', lw=2, ls='-',
                   label=f'Onset  frame {onset_idx}')
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_xlabel('Frame index')
    ax.set_ylabel('FFT fringe-band energy (log)')
    ax.set_title('FFT fringe-band energy across Field frames')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)
    ticks = list(range(n))
    xlabels = [f.split('_')[-1].replace('.bmp', '')[-6:] for f in field_files]
    ax.set_xticks(ticks)
    ax.set_xticklabels(xlabels, rotation=45, ha='right', fontsize=7)

    ax2 = axes[1]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(psd_store)))
    for ci, idx in enumerate(sorted(psd_store)):
        _, radial, r_vals, *_ = psd_store[idx]
        ts_label = field_files[idx].split('_')[-1].replace('.bmp', '')[-6:]
        lbl = f'frame {idx} ({ts_label})'
        if idx == onset_idx:
            lbl += ' ← onset'
        ax2.semilogy(r_vals_ref[:len(radial)], radial + 1,
                     color=colors[ci], lw=1.5, label=lbl)
    ax2.axvspan(f_lo_ref, f_hi_ref, alpha=0.12, color='cyan', label='Fringe band')
    ax2.set_xlim(0, min(200, len(r_vals_ref)))
    ax2.set_xlabel('Radial frequency (cycles / ROI)')
    ax2.set_ylabel('Radial PSD (log)')
    ax2.set_title('Radial power spectrum (key frames)')
    ax2.legend(fontsize=8)
    ax2.grid(True, which='both', alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, 'fft_energy_plot.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[{_ts()}] Saved {out_path}")

    print(f"\n[{_ts()}] Done.  Total elapsed: {time.time()-t_total:.2f}s")

    if onset_idx is not None:
        onset_ts = field_files[onset_idx].split('_')[-1].replace('.bmp', '')
        print(f"\n  RESULT  Frame {onset_idx} ({onset_ts}) — "
              f"{energies[onset_idx]/base_mean:.1f}× baseline, first crossing")


if __name__ == '__main__':
    main()
