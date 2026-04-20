import numpy as np
import cv2

def fast_radial_symmetry(img, radii, alpha=2, std_dev=0.5):
    """
    FRS 算法实现
    :param img: 输入灰度图 (float32)
    :param radii: 搜索的半径列表 (例如 [10, 20, 30])，对应条纹可能的尺寸
    :param alpha: 径向严格度因子，通常取 2
    :param std_dev: 高斯平滑的标准差
    :return: 最终的对称性图 S
    """
    rows, cols = img.shape
    S = np.zeros((rows, cols), dtype=np.float32)
    
    # 1. 计算梯度
    gy, gx = np.gradient(img)
    mag = np.sqrt(gx**2 + gy**2)
    
    # 阈值处理：忽略极小梯度以抑制 30dB 噪声
    threshold = np.max(mag) * 0.1
    mask = mag > threshold
    
    # 归一化梯度向量
    gx = np.divide(gx, mag, out=np.zeros_like(gx), where=mask)
    gy = np.divide(gy, mag, out=np.zeros_like(gy), where=mask)

    for n in radii:
        # 2. 构造方向图 (Orientation Projection) 和 幅度图 (Magnitude Projection)
        O_n = np.zeros((rows, cols), dtype=np.float32)
        M_n = np.zeros((rows, cols), dtype=np.float32)
        
        # 遍历所有具有显著梯度的像素点
        y_indices, x_indices = np.where(mask)
        
        for y, x in zip(y_indices, x_indices):
            # 计算沿梯度方向距离为 n 的正负投影点
            # p+ 对应于暗中心（条纹凹处），p- 对应于亮中心（条纹凸处）
            for sign in [1, -1]:
                pos_x = int(round(x + sign * n * gx[y, x]))
                pos_y = int(round(y + sign * n * gy[y, x]))
                
                if 0 <= pos_x < cols and 0 <= pos_y < rows:
                    O_n[pos_y, pos_x] += sign
                    M_n[pos_y, pos_x] += sign * mag[y, x]

        # 3. 归一化并计算单半径下的对称性响应
        # 这里的 kn 是为了平衡不同半径下的权重
        kn = 9.9  # 经验常数
        O_n = np.clip(O_n / kn, -1, 1)
        M_n = M_n / kn
        
        F_n = np.sign(O_n) * (np.abs(O_n)**alpha) * np.abs(M_n)
        
        # 4. 高斯平滑，扩散投票能量
        S += cv2.GaussianBlur(F_n, (0, 0), sigmaX=std_dev * n)

    return S / len(radii)

def find_robust_centroid(symmetry_map, quantile=0.98):
    """
    通过二值化和连通域提取稳健的质心
    :param symmetry_map: FRS 输出的响应图
    :param quantile: 用于二值化的分位数阈值，越高越严格
    :return: (cx, cy) 质心坐标
    """
    # 1. 归一化到 0-255
    s_norm = cv2.normalize(symmetry_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 2. 自适应二值化：取高响应区域
    # 这里使用百分比阈值（例如前 2% 的高响应点），比固定阈值更稳健
    thresh_val = np.percentile(s_norm, quantile * 100)
    _, binary = cv2.threshold(s_norm, thresh_val, 255, cv2.THRESH_BINARY)
    
    # 3. 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    
    if num_labels <= 1:
        return None # 未发现有效中心
        
    # 4. 筛选最大面积的连通域（排除孤立噪声点）
    # stats 的索引 0 是背景，所以从 1 开始
    max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    
    # 5. 获取该连通域的质心
    cx, cy = centroids[max_label]
    
    return (cx, cy), binary

import numpy as np
from PIL import Image

def has_fringes(img_array, roi_slice=np.s_[1100:2100, 2000:3100],
                freq_band=(0.01, 0.15), power_threshold=0.02):
    """
    Returns True if periodic fringe structure is detected in the ROI.
    
    freq_band: normalized spatial frequencies to check (fringe spacing ~7-100px)
    power_threshold: fraction of total power that must be in the band
    """
    roi = img_array.astype(np.float32)
    # roi = img_array[roi_slice].astype(np.float32)
    roi -= roi.mean()
    
    # 2D FFT, shift DC to center
    F = np.fft.fftshift(np.fft.fft2(roi))
    power = np.abs(F) ** 2
    
    # Build radial frequency grid
    h, w = roi.shape
    fy = np.fft.fftshift(np.fft.fftfreq(h))
    fx = np.fft.fftshift(np.fft.fftfreq(w))
    FX, FY = np.meshgrid(fx, fy)
    freq_r = np.sqrt(FX**2 + FY**2)
    
    # Exclude DC (center)
    mask = (freq_r > freq_band[0]) & (freq_r < freq_band[1])
    band_power = power[mask].sum() / power.sum()
    
    return band_power > power_threshold, band_power

def main():
    
    import os
    import matplotlib.pyplot as plt

    sequance = []

    image_folder = './20260331_142456' # list2' # '../20260311ya/20260330_215619'
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith('.png')])
    for img_file in image_files:
        img_name = os.path.join(image_folder, img_file)
        base_filename = os.path.splitext(os.path.basename(img_name))[0]
        i = int(base_filename[-3:])

        print(f'start processing {i}')
        image = (cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)[1100:2100, 2000:3100]).astype(np.float32)
        
        if len(sequance) < 3:
            sequance.append(image)
            print(f'len(sequance): {len(sequance)}')

        elif len(sequance) == 3:
            sequance.append(image)

            print(f'start calculating dark_avg and flat_avg')
            # 1. 加载预先计算好的平均场（建议用 float32 存储以防溢出）
            dark_avg = np.zeros_like(sequance[0]).astype(np.float32)
            flat_avg = np.mean(np.array(sequance), axis=0).astype(np.float32)

            # 2. 计算校正系数 (Flat - Dark) 的全局平均值
            diff = flat_avg - dark_avg
            mean_diff = np.mean(diff)
            

        else:
            print(f'start correcting image')

            raw_f = image.astype(np.float32)
            # 核心公式：校正非均匀性
            corrected = (raw_f - dark_avg) * (mean_diff / (diff + 1e-6))
            img_pre = np.clip(corrected, 0, 255).astype(np.uint8)
            # INSERT_YOUR_CODE
            # Save img_pre with filename starting with img_file
            save_name = f"{img_file}_img_pre.png"
            cv2.imwrite(save_name, img_pre)
   
            flag, band_power = has_fringes(img_pre, roi_slice=np.s_[1100:2100, 2000:3100],
                freq_band=(0.01, 0.15), power_threshold=0.02)
            print(f'flag: {flag}, band_power: {band_power}')

            plt.figure()
            plt.subplot(121)
            plt.imshow(image, cmap='gray')
            plt.subplot(122)
            plt.imshow(img_pre, cmap='gray')
            plt.title(f'{img_file}')
            plt.show()

            # import contact_line_info as cli
            

            # enhanced_img, candidates = cli.extract_innermost_dark_boundary_v2(img_pre, area_threshold=0)

            # print(f'candidates: {candidates}')
  
        # # image = cv2.imread('fringe_image.tif', 0).astype(np.float32)
        # denoised = cv2.GaussianBlur(image, (5, 5), 1.5)

        # # 2. 执行 FRS (假设条纹环绕的半径大约在 50 到 150 像素之间)
        # radii_to_search = range(50, 151, 10)
        # s_map = fast_radial_symmetry(denoised, radii_to_search)


        # # 获取稳健质心
        # result = find_robust_centroid(s_map)

        # if result:
        #     (center_x, center_y), binary_mask = result
        #     print(f"稳健中心坐标: ({center_x:.2f}, {center_y:.2f})")
            
        #     # 可视化检查
        #     res_vis = cv2.cvtColor((s_map/s_map.max()*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        #     plt.figure()
        #     plt.scatter(center_x, center_y, color='red', s=100)
        #     plt.imshow(res_vis, cmap='gray')
        #     plt.title("Centroid Detection")
        #     plt.show()

        # # 3. 寻找最大值点（即中心）
        # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(symmetry_map)
        # print(f"检测到的中心坐标: {max_loc}")
        # plt.figure()
        # plt.imshow(symmetry_map, cmap='gray')
        # plt.title("symmetry_map")
        # plt.show()

if __name__ == '__main__':
    main()