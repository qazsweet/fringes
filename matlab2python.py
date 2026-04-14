import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from skimage import morphology, filters, feature
from PIL import Image
from skimage import morphology, measure
from scipy.interpolate import interp1d
from scipy.io import savemat

# =========================================================================
# 基于条纹骨架法的干涉图样相位解析与面型重建
# =========================================================================

## 0. 参数设置 (Parameter Initialization)
lambda_val = 460e-6  # 光源波长 (mm)
grid_res = 0.5       # 最终面型空间分辨率 (mm)
pixel_pitch = 0.01   # 每个像素代表的实际物理尺寸 (mm)

ROI_SLICE = np.s_[1200:2100, 2000:3100]
MIN_RING_PX = 80         # ignore fringe fragments shorter than this

# ── Load & crop ───────────────────────────────────────────────────────────────
img_full = np.array(Image.open('20260330_215719_341.png').convert('L'))
img_gray = img_full[ROI_SLICE]

img_double = img_gray.astype(float) / 255.0

# # 使用 Pillow 进行缩放（不依赖 OpenCV）
# img_double = np.array(
#     Image.fromarray((img_double * 255).astype(np.uint8)).resize((100, 100), resample=Image.BILINEAR)
# ) / 255.0

roi = img_double

## 3. 各向异性扩散滤波 (Anisotropic Diffusion Filtering)
# 3.1 高斯预滤波
sigma_val = 2.0
img_gaussian = filters.gaussian(img_double, sigma=sigma_val)

# 3.2 各向异性扩散 (使用 skimage 的 denoise_nl_means 或 simple diffusion)
# Python 中常用的各向异性扩散可以使用 medfilt 或自定义，这里推荐 skimage 的 rolling_ball 或简单的 nlm
from skimage.restoration import denoise_tv_chambolle
img_filtered = denoise_tv_chambolle(img_gaussian, weight=0.1) # TV去噪与扩散效果类似

## 3. 背景归一化 (Background Normalization)
se_size = 50
se = morphology.disk(se_size)
I_max = morphology.dilation(img_filtered, se)
I_min = morphology.erosion(img_filtered, se)

denominator = I_max - I_min
denominator[denominator == 0] = 1e-12 # 避免除以0

# 归一化到 [-1, 1]
img_norm = 2 * (img_filtered - I_min) / denominator - 1

## 4. 条纹极值提取（骨架提取）
threshold = 0
BW = img_norm < threshold

# 去除孤立噪点 (Remove small objects)
BW_cleaned = morphology.remove_small_objects(BW, min_size=50)

# 形态学骨架细化 (Thinning)
skeleton = morphology.skeletonize(BW_cleaned)

plt.figure("Step 4: Skeleton")
plt.imshow(skeleton, cmap='gray')
plt.title("Fringe Skeleton")
plt.show(block=False)

# 假设 H, W 是图像的高度和宽度
H, W = skeleton.shape

## 5.1. 剔除照明边缘效应
margin = 15
# 创建边缘掩码：边缘区域为 True，内部为 False
border_mask = np.ones((H, W), dtype=bool)
border_mask[margin:H-margin, margin:W-margin] = False

# 连通域分析 (相当于 bwconncomp)
labels = measure.label(skeleton)
regions = measure.regionprops(labels)

skeleton_cleaned = skeleton.copy()
for reg in regions:
    # 获取当前连通域的所有像素坐标
    coords = reg.coords # 格式为 [[r1, c1], [r2, c2], ...]
    # 如果任何一个像素落在边缘掩码内，整条抹除
    if np.any(border_mask[coords[:, 0], coords[:, 1]]):
        skeleton_cleaned[coords[:, 0], coords[:, 1]] = 0

# 去除微小毛刺 (相当于 bwmorph spur)
# skimage 没有直接的 spur 10 次迭代，通常使用 thin 或简单的 remove_small_objects
skeleton_cleaned = morphology.thin(skeleton_cleaned)

## 5.2 骨架拓扑分析与二次曲线插补
# 重新获取清理后的连通域
labels = measure.label(skeleton_cleaned)
regions = measure.regionprops(labels)
skeleton_closed = np.zeros((H, W), dtype=bool)

for reg in regions:
    branch_mask = (labels == reg.label)
    
    # 寻找端点 (使用特殊的卷积核寻找只有一个邻居的像素)
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
    neighbor_count = cv2.filter2D(branch_mask.astype(np.uint8), -1, kernel)
    ey, ex = np.where((neighbor_count == 11) & branch_mask) # 10(中心) + 1(邻居)
    
    branch_length = reg.area
    
    if len(ex) == 0:
        # 情况 A：闭合环
        skeleton_closed |= branch_mask
        
    elif len(ex) == 2:
        # 情况 B：开口线段
        ep_dist = np.linalg.norm([ex[0]-ex[1], ey[0]-ey[1]])
        
        # 滤除规则
        if ep_dist < branch_length * 0.7 and branch_length > 10:
            by, bx = np.where(branch_mask)
            
            # 坐标系旋转以防止拟合斜率无穷大
            theta = np.arctan2(ey[1]-ey[0], ex[1]-ex[0])
            rx = bx * np.cos(theta) + by * np.sin(theta)
            ry = -bx * np.sin(theta) + by * np.cos(theta)
            
            # 拟合二次曲线 (y = ax^2 + bx + c)
            p = np.polyfit(rx, ry, 2)
            
            # 在端点间插值
            rex1 = ex[0]*np.cos(theta) + ey[0]*np.sin(theta)
            rex2 = ex[1]*np.cos(theta) + ey[1]*np.sin(theta)
            step = 0.5 if rex2 > rex1 else -0.5
            rx_interp = np.arange(rex1, rex2, step)
            ry_interp = np.polyval(p, rx_interp)
            
            # 逆旋转回原系
            x_i = np.round(rx_interp * np.cos(-theta) + ry_interp * np.sin(-theta)).astype(int)
            y_i = np.round(-rx_interp * np.sin(-theta) + ry_interp * np.cos(-theta)).astype(int)
            
            # 越界检查并写入
            valid = (x_i >= 0) & (x_i < W) & (y_i >= 0) & (y_i < H)
            branch_mask[y_i[valid], x_i[valid]] = True
            
            skeleton_closed |= branch_mask

# 再次细化确保单像素宽度
skeleton_closed = morphology.thin(skeleton_closed)



plt.figure("Step 4: Skeleton")
plt.imshow(skeleton_closed, cmap='gray')
plt.title("Fringe skeleton_closed")
plt.show(block=False)


## 5.3 独立提取面积与面心
from scipy.ndimage import binary_fill_holes
labels_final = measure.label(skeleton_closed)
regions_final = measure.regionprops(labels_final)
contour_list = []

for reg in regions_final:
    temp_skel = (labels_final == reg.label)
    # 填充孔洞 (相当于 imfill holes)
    filled_contour = binary_fill_holes(temp_skel)
    
    area_skel = reg.area
    area_filled = np.sum(filled_contour)
    
    if area_filled > area_skel * 1.5:
        # 记录中心和面积
        contour_list.append({
            'area': area_filled,
            'centroid': reg.centroid, # 注意：skimage 返回 (row, col) 即 (y, x)
            'coords': reg.coords
        })

if not contour_list:
    raise ValueError("未检测到任何有效封闭的等高线！")

## 5.4 排序与极值输出
# 按面积从小到大排序
contour_sorted = sorted(contour_list, key=lambda x: x['area'])

# 提取极值点 (最小环的中心)
extremum_y, extremum_x = contour_sorted[0]['centroid']
print(f"检测到干涉极值点坐标: ({extremum_x:.2f}, {extremum_y:.2f})")

y_center, x_center = contour_sorted[0]['centroid']

# 获取骨架像素坐标
row_skel, col_skel = np.where(skeleton_closed)

# # 计算物理距离
# dist_to_center = np.sqrt((col_skel - x_center)**2 + (row_skel - y_center)**2) * pixel_pitch

# # 简单的距离分层法分配级次
# max_dist = np.max(dist_to_center)
# num_bins = 50
# bins = np.linspace(0, max_dist, num_bins)
# bin_idx = np.digitize(dist_to_center, bins) - 1

# 根据每个骨架点所属的等高线环，分配级次 m
# 对于每个骨架像素，找到它属于 contour_sorted 的哪一个成员
bin_idx = np.zeros_like(row_skel, dtype=int)
label_map = labels_final[row_skel, col_skel]
for i, contour in enumerate(contour_sorted):
    # contour['coords'] 是该等高线内所有像素的位置
    mask = np.isin(label_map, labels_final[contour['coords'][:, 0], contour['coords'][:, 1]][0])
    bin_idx[mask] = i+1

# 计算高度 z = m * lambda / 2
z_skel = bin_idx * (lambda_val / 2.0)

## 6. 全场相位解包与曲面插值 (Surface Interpolation)
# 准备散点数据
points = np.column_stack((col_skel * pixel_pitch, row_skel * pixel_pitch))

# 生成目标网格 - 缩小为原始 ROI 的 1/10
h, w = img_double.shape
print(f'h:{h}, w:{w}')
x_phys = np.arange(0, w) * pixel_pitch
y_phys = np.arange(0, h) * pixel_pitch

# 目标分辨率（1/10 大小）
resize_factor = 0.1
w_small = int(w * resize_factor)
h_small = int(h * resize_factor)

# x_phys_small 和 y_phys_small 是插值的目标物理坐标（尽量覆盖原 ROI 范围）
x_phys_small = np.linspace(x_phys.min(), x_phys.max(), w_small)
y_phys_small = np.linspace(y_phys.min(), y_phys.max(), h_small)
grid_x, grid_y = np.meshgrid(x_phys_small, y_phys_small)

# 使用 griddata 进行插值 (cubic 对应 natural)
# 只对 z_skel 非零的位置进行插值
nonzero_mask = z_skel != 0
z_surface = griddata(points[nonzero_mask], z_skel[nonzero_mask], (grid_x, grid_y), method='cubic')

print(f'z_surface shape: {z_surface.shape}')
savemat('data.mat', {'x_center': x_center, 'y_center': y_center, 'z_surface': z_surface})

# ## 7. 最终结果输出 (Output)
# fig = plt.figure("Step 7: 3D Surface Reconstruction")
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(grid_x, grid_y, z_surface, cmap='jet', edgecolor='none')
# fig.colorbar(surf)
# ax.set_title(f"Reconstructed Surface (Res: {grid_res} mm)")
# ax.set_xlabel('X (mm)')
# ax.set_ylabel('Y (mm)')
# ax.set_zlabel('Height (mm)')


plt.figure()
plt.imshow(z_surface, cmap='viridis')
plt.title("Reconstructed Surface")
plt.show()