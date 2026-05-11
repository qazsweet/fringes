"""
批量处理：遍历目录下图片 → npy 平场矫正 → ROI + FFT 带通（与 contact_line_info_r 一致）
→ 缩小 1/4 → 提取边界并在图上绘制 → 保存。

平场：使用 mean_dark.npy、divisor.npy（与 nothing2first_fringes 保存格式一致），
尺寸与图像不一致时会双线性缩放到当前图尺寸。
默认标定为本脚本所在目录下的 mean_dark.npy、divisor.npy（例如 E:\\program\\20260430scimage\\…）。

默认输入目录为本仓库下 PJ_20260505155312/imprint；无参数时输出到同级的 imprint_fft_edge_out。

每张图处理后会弹出 matplotlib 2×4 子图展示各步结果（关闭窗口后继续下一张）；
加 --no-show-steps 可关闭弹窗，仅写文件。

示例：
  python batch_roi_fft_edge.py
  python batch_roi_fft_edge.py --no-show-steps
  python batch_roi_fft_edge.py --no-ffc

  python batch_roi_fft_edge.py ./imprint ./out_edges ^
    --mean-dark-npy ./mean_dark.npy --divisor-npy ./divisor.npy

  python batch_roi_fft_edge.py ./imprint ./out_edges --calib-dir ./calib
  （在 calib 目录下查找 mean_dark.npy 与 divisor.npy）
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

SCRIPT_DIR = Path(__file__).resolve().parent

# 默认与 batch_contact_line_imprint 一致：PJ_20260505155312/imprint（即 E:\\program\\20260430scimage\\...）
DEFAULT_IMPRINT_DIR = SCRIPT_DIR / "field0429ij2"   #"PJ_20260505155312" / "imprint"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "field0429ij2" /  "imprint_fft_edge_out" # "PJ_20260505155312" / "imprint_fft_edge_out"
DEFAULT_MEAN_DARK_NPY = SCRIPT_DIR / "mean_dark.npy"
DEFAULT_DIVISOR_NPY = SCRIPT_DIR / "divisor.npy"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# 与 contact_line_info_r.py / nothing2first_fringes.py 一致
ROI_SLICE = np.s_[130:2700, 1050:3800] 
ROI_SLICE2 = np.s_[:, :] # -300[130:2800, 1234:3500]

# 频域掩模保留区域（针对该 ROI 分辨率）
MASK_Y0, MASK_Y1 = 1200, 1375
MASK_X0, MASK_X1 = 1275, 1475

COLOR_EDGE = (0, 255, 0)  # BGR 绿色

# Pre-contact detection uses central-disk standard deviation on the
# preprocessed (CLAHE) quarter-resolution image.  In pre-contact frames the
# image centre contains visible Newton's ring texture (std ≈ 20–55); once
# contact forms, the centre is a flat dark zone with no fringe pattern
# (std ≤ 11 even for the very first contact frame).
# Threshold sits in the gap: pre-contact min ≈ 20, contact max ≈ 11.
CENTRAL_DISK_R    = 50    # radius of the test disk (quarter-res pixels)
CENTRAL_STD_THRESH = 15   # std > this → ring texture present → pre-contact


def iter_images(folder: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            out.append(p)
    return out


def load_gray(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        
    if img is None:
        raise ValueError(f"无法读取图像: {path}")
    return img


def load_ffc_arrays(
    mean_dark_path: Path, divisor_path: Path, target_hw: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    mean_dark = np.load(str(mean_dark_path)).astype(np.float32)
    divisor = np.load(str(divisor_path)).astype(np.float32)
    th, tw = target_hw
    if mean_dark.shape != (th, tw):
        mean_dark = cv2.resize(mean_dark, (tw, th), interpolation=cv2.INTER_AREA).astype(
            np.float32
        )
        divisor = cv2.resize(divisor, (tw, th), interpolation=cv2.INTER_AREA).astype(
            np.float32
        )
        divisor = np.where(divisor < 1.0, 1.0, divisor)
        divisor = divisor / float(divisor.mean())
    return mean_dark, divisor

def flat_field_correct(raw_f32, mean_dark, divisor):
    corr = (raw_f32 - mean_dark) / divisor
    return np.clip(corr, 0, None)

def flat_field_to_uint8(
    gray_u8: np.ndarray,
    mean_dark: np.ndarray,
    divisor: np.ndarray,
) -> np.ndarray:
    raw_f32 = gray_u8.astype(np.float32)
    corr = flat_field_correct(raw_f32, mean_dark, divisor)
    lo = float(np.min(corr))
    hi = float(np.max(corr))
    if hi <= lo:
        return np.zeros_like(gray_u8, dtype=np.uint8)
    return np.clip((corr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def fft_mask_filter_roi(roi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    与 contact_line_info_r.py 第 17–33 行等价的带通：仅保留 shift 频谱中矩形区域。
    返回 (img_back float, img_filtered uint8, dft_shift, fshift)。
    """
    rows, cols = roi.shape
    dft = np.fft.fft2(roi.astype(np.float32))
    dft_shift = np.fft.fftshift(dft)

    mask = np.zeros((rows, cols), dtype=np.float32)
    y0 = max(0, min(MASK_Y0, rows))
    y1 = max(0, min(MASK_Y1, rows))
    x0 = max(0, min(MASK_X0, cols))
    x1 = max(0, min(MASK_X1, cols))
    if y1 > y0 and x1 > x0:
        yc = (y0 + y1) // 2
        xc = (x0 + x1) // 2
        r = min((y1 - y0), (x1 - x0)) // 2
        Y, X = np.ogrid[y0:y1, x0:x1]
        # 原硬掩模：圆内 1、圆外 0
        # mask_in_box = ((Y - yc) ** 2 + (X - xc) ** 2) <= r**2
        # mask[y0:y1, x0:x1][mask_in_box.astype(np.bool_)] = 1.0
        dist = np.sqrt(((Y - yc).astype(np.float32) ** 2 + (X - xc).astype(np.float32) ** 2))
        dr = np.maximum(dist - float(r), 0.0)
        sigma = max(float(r) * 0.35, 1.0)
        mask[y0:y1, x0:x1] = np.exp(-(dr**2) / (2.0 * sigma**2))
        # mask[1275:1295, :] = 1
        # mask[:, 1355:1390] = 1
    
    d0=10; rl=0.5; rh=2.0; c=1
    crow, ccol = rows//2 , cols//2
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    d2 = x*x + y*y
    lp = 1 - np.exp(-c * (d2 / (d0**2)))
    h = (rh - rl) * lp + rl
    
    fshift = dft_shift * h.astype(np.float32)

    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.abs(np.fft.ifft2(f_ishift))
    img_filtered = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img_back, img_filtered, dft_shift, fshift


def log_magnitude_spectrum(z: np.ndarray) -> np.ndarray:
    """与 contact_line_info_r 一致：20*log(|·|+1)"""
    return 20 * np.log(np.abs(z) + 1)


def plot_processing_steps(
    title: str,
    roi_raw: np.ndarray,
    roi_ffc: np.ndarray,
    dft_shift: np.ndarray,
    fshift: np.ndarray,
    img_filtered: np.ndarray,
    small: np.ndarray,
    edges: np.ndarray,
    vis_bgr: np.ndarray,
    *,
    use_ffc: bool,
) -> None:
    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    ax = axes.ravel()
    panels: list[tuple[str, np.ndarray, str | None]] = [
        ("1 原图 ROI", roi_raw, "gray"),
        ("2 平场后 ROI" if use_ffc else "2 未平场（同左）", roi_ffc, "gray"),
        ("3 log|fftshift|", log_magnitude_spectrum(dft_shift), "gray"),
        ("4 log|掩模后频谱|", log_magnitude_spectrum(fshift), "gray"),
        ("5 IFFT 重建", img_filtered, "gray"),
        ("6 缩小 1/4", small, "gray"),
        ("7 Canny 边缘", edges, "gray"),
        ("8 边界叠加", vis_rgb, None),
    ]
    for i, (ti, data, cmap) in enumerate(panels):
        ax[i].set_title(ti, fontsize=10)
        if cmap:
            ax[i].imshow(data, cmap=cmap)
        else:
            ax[i].imshow(data)
        ax[i].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def crop_roi_safe(gray: np.ndarray) -> np.ndarray | None:
    """按 ROI_SLICE 裁剪；若图像小于 ROI 则返回 None。"""
    h, w = gray.shape
    ys, ye = ROI_SLICE[0].start, ROI_SLICE[0].stop
    xs, xe = ROI_SLICE[1].start, ROI_SLICE[1].stop
    if ye > h or xe > w or ys < 0 or xs < 0:
        return None
    return gray[ROI_SLICE].copy()

def auto_gamma_correction(image, target_mean=128):
        current_mean = np.mean(image)
        if current_mean <= 0:
            return image
        gamma = np.log(target_mean / 255.0) / np.log(current_mean / 255.0)
        gamma = np.clip(gamma, 0.25, 4.0)
        invGamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")
        return cv2.LUT(image, table)

def preprocess_img( original_img):
    img = cv2.resize(
        original_img,
        (original_img.shape[1] // 2, original_img.shape[0] // 2),
        interpolation=cv2.INTER_AREA,
    )
    img_gamma = auto_gamma_correction(img)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(img_gamma)
    enhanced_img = cv2.resize(
        enhanced_img,
        (enhanced_img.shape[1] // 2, enhanced_img.shape[0] // 2),
        interpolation=cv2.INTER_AREA,
    )
    return enhanced_img

def quarter_and_edges(img_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray, list]:
    h, w = img_u8.shape
    # small = cv2.resize(img_u8, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)
    small = preprocess_img(img_u8)
    # small = img_u8
    
    # U, s, Vh = np.linalg.svd(img_u8 / 255.0, full_matrices=False)
    # start_k=3
    # end_k=250
    # # 构造一个只保留中段奇异值的对角矩阵
    # # 丢弃前 start_k 个（去掉大背景）
    # # 丢弃 end_k 之后的（去掉噪声）
    # s_mask = np.zeros_like(s)
    # s_mask[start_k:end_k] = s[start_k:end_k]
    
    # img_contour = np.dot(U, np.dot(np.diag(s_mask), Vh))
    # small = np.clip(img_contour * 255, 0, 255).astype(np.uint8)

    # blurred = cv2.GaussianBlur(small, (3, 3), 0)
    # med = float(np.median(blurred))
    # lo = max(0.0, 0.66 * med)
    # hi = min(255.0, 1.33 * med)
    # if hi <= lo:
    #     hi = lo + 1.0
    # edges = cv2.Canny(blurred, int(lo), int(hi))
    # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    blurred = cv2.medianBlur(small, 7)
    blurred = cv2.GaussianBlur(blurred, (9, 9), 0)
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        55,
        3,
    )
    kernel = np.ones((3, 3), np.uint8)
    kernelm = np.ones((5, 5), np.uint8)
    kernell = np.ones((7, 7), np.uint8)
    edges = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernelm)
    edges2 = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernelm)

    # # 在 edges2 上用霍夫圆变换检测圆
    # circles = cv2.HoughCircles(
    #     edges2,
    #     cv2.HOUGH_GRADIENT,
    #     dp=1.2,
    #     minDist=min(h, w) // 8,
    #     param1=50,
    #     param2=30,
    #     minRadius=min(h, w) // 10,
    #     maxRadius=min(h, w) // 2,
    # )
    # hough_circle_contours = []
    
    # edges2 = cv2.morphologyEx(edges2, cv2.MORPH_CLOSE, kernell)
    # 查找中心实心形状（面积最大且接近圆的轮廓）
    contours, _ = cv2.findContours(
        edges2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    return small, edges, contours


def _fit_circle_kasa(pts: np.ndarray):
    """Algebraic (Kasa) least-squares circle fit on (N,2) array. Returns (cx,cy,r) or None."""
    if len(pts) < 5:
        return None
    x = pts[:, 0].astype(np.float64)
    y = pts[:, 1].astype(np.float64)
    A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
    b = x ** 2 + y ** 2
    try:
        res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        print(f'np.linalg.LinAlgError:{np.linalg.lstsq(A, b, rcond=None)}')
        return None
    cx, cy, c = res
    r2 = c + cx ** 2 + cy ** 2
    print(f'_fit_circle_kasa: {cx}, {cy}, {np.sqrt(r2)}')
    return (float(cx), float(cy), float(np.sqrt(r2))) if r2 > 0 else None


def _clamp_circle_to_image_bounds(
    kx: float, ky: float, kr: float, w: float, h: float
) -> tuple[float, float, float]:
    """
    Keep the fitted circle inside the image rectangle and cap diameter at the
    diagonal length: r <= min(half_diagonal, distance to each image edge).
    Center is clipped to [0, w] x [0, h] (same convention as corner_defs).
    """
    kx_c = float(np.clip(kx, 0.0, w))
    ky_c = float(np.clip(ky, 0.0, h))
    half_diag = 0.5 * float(np.hypot(w, h))
    # diag_limit = np.sqrt(w**2 + h**2)/2
    # r_inscribed = min(kx_c, ky_c, w - kx_c, h - ky_c)
    kr_c = min(float(kr), half_diag)
    return kx_c, ky_c, kr_c


def draw_edges_bgr(
    small_gray: np.ndarray, contours: list, img_name: str = ""
) -> tuple[np.ndarray, float, str]:
    """
    Detect fringe ring and return (vis_bgr, radius_px, flag).

    flag values
    -----------
    ""            – valid detection; radius_px is the innermost contact ring radius
    "pre_contact" – Newton's rings from a pre-contact (no large contact zone) phase;
                    ring texture is visible around the image centre (centre disk std >
                    CENTRAL_STD_THRESH), meaning complete concentric rings are still
                    present.  radius_px is returned as 0.0.
    "no_detection"– no reliable fringe ring found; radius_px = 0.0.
    """
    vis = cv2.cvtColor(small_gray, cv2.COLOR_GRAY2BGR)
    H_vis, W_vis = vis.shape[:2]
    img_cx, img_cy = W_vis / 2.0, H_vis / 2.0

    # ── Pre-contact detection: central disk std ───────────────────────────────
    # Measure ring texture in a disk around the image centre.  Pre-contact frames
    # have complete Newton's rings visible near the centre (std ≈ 20–55 in the
    # CLAHE-enhanced image).  Once contact forms the centre is a flat dark zone
    # (std ≤ 11 even for the very first contact frame).  This check runs before
    # contour analysis so it triggers even when contour detection is unreliable.
    yy, xx = np.mgrid[0:H_vis, 0:W_vis]
    disk_mask = (xx - img_cx) ** 2 + (yy - img_cy) ** 2 <= CENTRAL_DISK_R ** 2
    central_std = float(small_gray[disk_mask].astype(np.float64).std())
    if central_std > CENTRAL_STD_THRESH:
        # Draw all contours in orange to visualise the pre-contact ring pattern
        cv2.drawContours(vis, contours, -1, (0, 165, 255), 1)
        cv2.putText(vis, f"PRE-CONTACT r=0  (std={central_std:.1f})",
                    (max(0, W_vis // 2 - 200), H_vis // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        print(f"  [PRE-CONTACT]  {img_name}  central_std={central_std:.1f}  → r=0")
        return vis, 0.0, "pre_contact"

    cnt_infos = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
        else:
            cx, cy = 0.0, 0.0
        area = cv2.contourArea(cnt)
        (circle_cx, circle_cy), fit_radius = cv2.minEnclosingCircle(cnt)
        cnt_infos.append({
            "cnt": cnt,
            "cx": cx,
            "cy": cy,
            "area": area,
            "fit_radius": float(fit_radius)
        })

    detected = False

    # ── DBSCAN path: works when complete fringe rings form closed contours ────
    if len(cnt_infos) > 0:
        cxs = np.array([(ci["cx"], ci["cy"]) for ci in cnt_infos])
        from sklearn.cluster import DBSCAN
        db = DBSCAN(eps=10.0, min_samples=2).fit(cxs)
        labels = db.labels_
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
        if len(counts) > 0:
            # Among same-size clusters break ties by distance of cluster centroid
            # to image centre — fringe rings cluster near centre, corner noise does not.
            max_count = counts.max()
            candidate_lbls = unique_labels[counts == max_count]
            def _cluster_dist(lbl):
                mi = np.where(labels == lbl)[0]
                return np.hypot(np.mean([cnt_infos[i]["cx"] for i in mi]) - img_cx,
                                np.mean([cnt_infos[i]["cy"] for i in mi]) - img_cy)
            max_lbl = min(candidate_lbls, key=_cluster_dist)
            idxs = np.where(labels == max_lbl)[0]
            if len(idxs) > 1:
                for i in idxs:
                    cv2.drawContours(vis, [cnt_infos[i]["cnt"]], -1, (255, 0, 0), 2)

            large_area_idxs = [i for i in idxs if cnt_infos[i]["fit_radius"] > 40]
            if large_area_idxs:
                # min_idx = min(large_area_idxs, key=lambda i: cnt_infos[i]["fit_radius"])
                
                
                # 选第二小的
                fit_radii = sorted([(cnt_infos[i]["fit_radius"], i) for i in large_area_idxs])
                if len(fit_radii) >= 2:
                    min_idx = fit_radii[0][1]
                else:
                    min_idx = fit_radii[0][1]
         


                r  = cnt_infos[min_idx]["fit_radius"]
                cx_r = int(cnt_infos[min_idx]["cx"])
                cy_r = int(cnt_infos[min_idx]["cy"])

                cv2.drawContours(vis, [cnt_infos[min_idx]["cnt"]], -1, (0, 0, 255), 2)
                cv2.circle(vis, (cx_r, cy_r), int(r), (0, 255, 0), 2)
                cv2.circle(vis, (cx_r, cy_r), 3, (0, 255, 0), 1)
                cv2.putText(vis, f"r={r:.0f}", (cx_r - 40, cy_r),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                print(f"  [DBSCAN]  {img_name}  r={r:.1f}  cx={cx_r}  cy={cy_r}")
                detected = True

                # for cnt in  contours:
                #     cv2.drawContours(vis, [cnt], -1, (128, 128, 0), 2)
                return vis, float(r), ""

    # ── Kasa fallback: contact area fills the image, only corner/edge arcs visible ─
    # For each image corner select the arc whose centroid is closest to that corner
    # (= the innermost fringe just outside the contact edge at that corner).
    # Combining arcs from ≥2 corners gives a robust algebraic circle fit even when
    # the full contact boundary circle is never visible as a single closed contour.
    if not detected and len(cnt_infos) > 0:
        MIN_ARC_AREA = 200   # ignore tiny noise fragments

        # One entry per corner: (corner_row, corner_col)
        corner_defs = [(0, 0), (0, W_vis), (H_vis, 0), (H_vis, W_vis)]
        corner_pts_list = []
        corner_cnts_list = []   # parallel list of contours for drawing
        corner_defs_matched = []  # (row, col) of image corner for each arc in corner_pts_list
        d_center_min = min(H_vis, W_vis) / 4   # excludes rogue inner contours
        for cr, cc in corner_defs:
            # Candidates in the same quadrant as this corner with area > threshold
            # and centroid far enough from image centre (excludes rogue noise
            # fragments inside the dark contact zone, which have small d_center).
            candidates = [
                i for i, ci in enumerate(cnt_infos)
                if ci["area"] > MIN_ARC_AREA
                and ((ci["cx"] < W_vis / 2) == (cc == 0))
                and ((ci["cy"] < H_vis / 2) == (cr == 0))
                and np.hypot(ci["cx"] - img_cx, ci["cy"] - img_cy) > d_center_min
            ]
            if candidates:
                # Contact boundary = the arc with the LARGEST area among the
                # filtered candidates.  After excluding centroids close to the
                # image centre the innermost contact ring is always the winner:
                #   Real images: one dominant arc per corner; the d_center filter
                #     eliminates rogue inner contours (e.g. BL area=36226 d_c=105).
                #   Synthetic (multiple rings at corners): all legitimate rings
                #     have d_center >> d_center_min; the innermost ring
                #     (contact boundary) subtends the widest angle → largest area.
                best_i = max(candidates, key=lambda i: cnt_infos[i]["area"])
                corner_pts_list.append(contours[best_i].reshape(-1, 2))
                corner_cnts_list.append(contours[best_i])
                corner_defs_matched.append((cr, cc))

        # Need arcs from at least 2 corners for a stable fit
        if len(corner_pts_list) > 2:
            combined = np.vstack(corner_pts_list)
            result = _fit_circle_kasa(combined)
        elif len(corner_pts_list) == 2:
            print(f'Only two corner arcs: Kasa-fit using both arcs plus remaining two corners')
            # 获取已匹配的corner定义
            matched_corners = set(corner_defs_matched)
            # 找出未匹配的两个角
            remaining_corners = [xy for xy in corner_defs if xy not in matched_corners]
            # 转换为 (x=cc, y=cr) 的坐标，注意顺序！
            other_corners_xy = np.array(
                [[float(cc), float(cr)] for cr, cc in remaining_corners],
                dtype=np.float64,
            )
            # 从每个 corner arc 各选2个点，再和 other_corners_xy 合并为 combined
            # 从每个corner arc计算一个点的平均值，再和other_corners_xy合并为combined
            selected_points = []
            for arc in corner_pts_list:
                avg_pt = np.mean(arc, axis=0)
                selected_points.append(avg_pt)
            selected_points = np.array(selected_points, dtype=np.float64)  # shape (2,2)
     
            combined = np.vstack([selected_points, other_corners_xy])
     
            mean_point = np.mean(combined, axis=0)
            
            cw_vis = mean_point[0] - selected_points[0][0] 
            ch_vis = mean_point[1]- selected_points[0][1] 
            half_diag = float(np.hypot(cw_vis, ch_vis))
            print(f'cw_vis: {cw_vis}, ch_vis: {ch_vis}')
            result = float(mean_point[0]), float(mean_point[1]), half_diag

            # result = _fit_circle_kasa(combined)
 
        elif len(corner_pts_list) == 1:
            print(f'Only one corner arc: Kasa-fit using that arc plus the other three image corners')
            # Only one corner arc: Kasa-fit using that arc plus the other three image corners
            # (same (x,y) as corner_defs: x=col=cc, y=row=cr).
            cr0, cc0 = corner_defs_matched[0]
            other_corners_xy = np.array(
                [[float(cc), float(cr)] for cr, cc in corner_defs if (cr, cc) != (cr0, cc0)],
                dtype=np.float64,
            )

            # 求 corner_pts_list[0] 中点的平均值
            corner_cr, corner_cc = corner_defs_matched[0]
            avg_pt = np.mean(corner_pts_list[0], axis=0)
            print(f'Average point of corner_pts_list[0]: {avg_pt}')
            # 求 other_corners_xy 和 avg_pt 的平均点
            # other_corners_xy: (3, 2)，avg_pt: (2,)
            points_to_average = np.vstack([other_corners_xy, avg_pt])
            mean_point = np.mean(points_to_average, axis=0)
            print(f'Average point of other_corners_xy and avg_pt: {mean_point}')

            cw_vis = mean_point[0] - avg_pt[0] 
            ch_vis = mean_point[1]- avg_pt[1] 
            half_diag = float(np.hypot(cw_vis, ch_vis))
            print(f'cw_vis: {cw_vis}, ch_vis: {ch_vis}')


            result = float(mean_point[0]), float(mean_point[1]), half_diag
            # # 计算 corner_pts_list[0] 内所有点到其所属角点的距离的统计量（如均值或中位数）
            # # corner_defs_matched[0] 是 (cr, cc) 格式（行,列）即 (y,x)
            # # corner_pts_list[0] 每个点是 (x, y) = (col, row)
            # corner_coords = np.array([corner_cc, corner_cr], dtype=np.float64)  # (x, y)
            # # 计算所有点到角的欧几里得距离
            # dists = np.linalg.norm(corner_pts_list[0] - corner_coords, axis=1)
            # # 取距离的中位数（也可以改为np.mean）
            # dist_stat = np.median(dists)
            # print(f'Median distance from corner_pts_list[0] to its corner {corner_coords}: {dist_stat}')
            # # 在 corner_pts_list[0] 中选一个中间位置的点
            # corner_arc = corner_pts_list[0]
            # mid_idx = len(corner_arc) // 2
            # mid_pt = corner_arc[mid_idx-1:mid_idx+1]
            # combined = np.vstack([mid_pt, other_corners_xy])
            # print(f'combined: {combined}')
            # result = _fit_circle_kasa(combined)
            
        else:
            result = None

        if result is not None:
            kx, ky, kr = result
            print(f'before clamp: {kx}, {ky}, {kr}')
            kx, ky, kr = _clamp_circle_to_image_bounds(
                kx, ky, kr, float(W_vis), float(H_vis)
            )
            print(f'after clamp: {kx}, {ky}, {kr}')
        if result is not None and kr > 40:
            # Draw the selected corner arcs in orange so they are visible
            cv2.drawContours(vis, corner_cnts_list, -1, (0, 165, 255), 2)
            # Cyan circle + asterisk suffix to distinguish Kasa fit from DBSCAN fit
            cv2.circle(vis, (int(kx), int(ky)), int(kr), (0, 255, 255), 2)
            cv2.circle(vis, (int(kx), int(ky)), 3, (0, 255, 255), -1)
            cv2.putText(vis, f"r={kr:.0f}*", (int(kx) - 40, int(ky)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 200), 2)
            n_corners = len(corner_pts_list)
            print(f"  [Kasa/{n_corners}c]  {img_name}  r={kr:.1f}  cx={kx:.1f}  cy={ky:.1f}")
            return vis, float(kr), ""

    print(f"  [NO-DETECT]  {img_name}")
    return vis, 0.0, "no_detection"


def _timing_step_order(use_ffc: bool) -> list[tuple[str, str]]:
    order: list[tuple[str, str]] = [("load_gray", "读取图像")]
    if use_ffc:
        order.append(("ffc_load_arrays", "加载平场系数（尺寸变化时重载）"))
        order.append(("ffc_apply", "平场校正"))
    order.extend(
        [
            ("crop_roi", "ROI 裁剪"),
            ("fft_filter", "FFT 带通"),
            ("quarter_edges", "缩小与边缘检测"),
            ("draw_edges", "轮廓/圆拟合绘制"),
            ("save", "保存结果图"),
        ]
    )
    return order


def _print_timing_for_image(
    img_name: str,
    timers: dict[str, float],
    *,
    use_ffc: bool,
    partial: bool = False,
) -> None:
    """打印单张图各步耗时（秒）；不含 matplotlib 弹窗。partial=True 时仅输出已记入 timers 的步骤。"""
    order = _timing_step_order(use_ffc)
    tag = "（仅已执行步骤）" if partial else ""
    print(f"\n--- [{img_name}] 耗时（不含画图 plt.show）{tag}---")
    total = 0.0
    for key, label in order:
        if key not in timers:
            continue
        t = float(timers[key])
        total += t
        print(f"  {label}: {t:.3f} s")
    print(f"  本图合计: {total:.3f} s")


def resolve_calib_paths(calib_dir: Path) -> tuple[Path, Path]:
    md = calib_dir / "mean_dark.npy"
    div = calib_dir / "divisor.npy"
    if not md.is_file():
        raise FileNotFoundError(f"缺少 {md}")
    if not div.is_file():
        raise FileNotFoundError(f"缺少 {div}")
    return md, div


def main() -> int:
    ap = argparse.ArgumentParser(description="ROI FFT + 1/4 缩小 + 边界可视化（批量）")
    ap.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_IMPRINT_DIR,
        help=f"输入图片目录（默认: {DEFAULT_IMPRINT_DIR}）",
    )
    ap.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录（默认: {DEFAULT_OUTPUT_DIR}）",
    )
    ap.add_argument(
        "--calib-dir",
        type=Path,
        default=None,
        help="含 mean_dark.npy 与 divisor.npy 的目录",
    )
    ap.add_argument(
        "--mean-dark-npy",
        type=Path,
        default=DEFAULT_MEAN_DARK_NPY,
        help=f"mean_dark.npy 路径（默认: {DEFAULT_MEAN_DARK_NPY}）",
    )
    ap.add_argument(
        "--divisor-npy",
        type=Path,
        default=DEFAULT_DIVISOR_NPY,
        help=f"divisor.npy 路径（默认: {DEFAULT_DIVISOR_NPY}）",
    )
    ap.add_argument(
        "--no-ffc",
        action="store_true",
        help="不平场矫正（忽略 npy）",
    )
    ap.add_argument(
        "--no-show-steps",
        dest="show_steps",
        action="store_false",
        help="不弹出 matplotlib 分步展示（适合批量无人值守）",
    )
    ap.set_defaults(show_steps=True)
    args = ap.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        print(f"输入目录不存在: {input_dir}", file=sys.stderr)
        return 1

    if args.calib_dir is not None:
        cdir = args.calib_dir.resolve()
        mean_dark_path, divisor_path = resolve_calib_paths(cdir)
    else:
        mean_dark_path = args.mean_dark_npy.resolve()
        divisor_path = args.divisor_npy.resolve()

    use_ffc = not args.no_ffc
    if use_ffc:
        if not mean_dark_path.is_file() or not divisor_path.is_file():
            print(
                "启用平场时未找到标定文件，请将 mean_dark.npy / divisor.npy 放到默认目录，"
                "或使用 --calib-dir / --mean-dark-npy / --divisor-npy 指定，或加 --no-ffc",
                file=sys.stderr,
            )
            print(f"  mean_dark: {mean_dark_path}", file=sys.stderr)
            print(f"  divisor:   {divisor_path}", file=sys.stderr)
            return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    paths = iter_images(input_dir)
    if not paths:
        print(f"目录中无支持的图片: {input_dir}", file=sys.stderr)
        return 1

    mean_dark: np.ndarray | None = None
    divisor: np.ndarray | None = None

    ok = 0
    skip = 0
    cached_shape: tuple[int, int] | None = None
    for path in paths:
        img_times: dict[str, float] = {}
        try:
            t0 = time.perf_counter()
            img = load_gray(path)
            gray = cv2.rotate(img, cv2.ROTATE_180)
            img_times["load_gray"] = time.perf_counter() - t0
            if use_ffc:
                if mean_dark is None or cached_shape != gray.shape:
                    t0 = time.perf_counter()
                    mean_dark, divisor = load_ffc_arrays(
                        mean_dark_path, divisor_path, gray.shape
                    )
                    img_times["ffc_load_arrays"] = time.perf_counter() - t0
                    cached_shape = gray.shape
                t0 = time.perf_counter()
                gray_work = flat_field_to_uint8(gray, mean_dark, divisor)
                img_times["ffc_apply"] = time.perf_counter() - t0
            else:
                gray_work = gray

            t0 = time.perf_counter()
            roi_raw = crop_roi_safe(gray)
            roi = crop_roi_safe(gray_work)
            img_times["crop_roi"] = time.perf_counter() - t0
            if roi is None or roi_raw is None:
                print(f"跳过（图像小于 ROI）: {path.name}", file=sys.stderr)
                _print_timing_for_image(path.name, img_times, use_ffc=use_ffc, partial=True)
                skip += 1
                continue

            t0 = time.perf_counter()
            _img_back, img_filtered, dft_shift, fshift = fft_mask_filter_roi(roi)
            img_times["fft_filter"] = time.perf_counter() - t0
            img_filtered_roi = img_filtered[ROI_SLICE2]
            t0 = time.perf_counter()
            small, edges, contours = quarter_and_edges(img_filtered_roi)
            img_small = cv2.resize(img_filtered_roi, (small.shape[1], small.shape[0]))
            img_times["quarter_edges"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            vis, radius_px, flag = draw_edges_bgr(img_small, contours, img_name=path.name)
            img_times["draw_edges"] = time.perf_counter() - t0

            if args.show_steps:
                plot_processing_steps(
                    path.name,
                    roi_raw,
                    roi,
                    dft_shift,
                    fshift,
                    img_filtered,
                    small,
                    edges,
                    vis,
                    use_ffc=use_ffc,
                )

            stem = path.stem
            out_vis = output_dir / f"{stem}_fft_edge.png"
            out_edge = output_dir / f"{stem}_edges_only.png"
            t0 = time.perf_counter()
            cv2.imwrite(str(out_vis), vis)
            cv2.imwrite(str(out_edge), edges)
            img_times["save"] = time.perf_counter() - t0
            _print_timing_for_image(path.name, img_times, use_ffc=use_ffc, partial=False)
            ok += 1
        except Exception as e:
            print(f"失败 {path.name}: {e}", file=sys.stderr)
            if img_times:
                _print_timing_for_image(path.name, img_times, use_ffc=use_ffc, partial=True)
            skip += 1

    print(f"完成: 成功 {ok}, 跳过/失败 {skip}, 输出目录 {output_dir}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())