import cv2
import numpy as np
import matplotlib.pyplot as plt


# =========================
# 输入路径
# =========================
img_path = "hi0120.png"


# =========================
# 读取高频图
# =========================
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# 转float
img = img.astype(np.float32)


# =========================
# 1. 去均值（突出高频变化）
# =========================
img_center = img - np.mean(img)


# =========================
# 2. 高频幅值放大
# =========================
gain = 5   # 可以改大，比如10、20

img_amp = img_center * gain


# =========================
# 3. 对称归一化
# 保留正负高频信息
# =========================
max_val = np.max(np.abs(img_amp))

img_norm = img_amp / max_val


# =========================
# 4. 映射到0-255
# =========================
img_vis = ((img_norm + 1) * 127.5).astype(np.uint8)



# =========================
# 5. CLAHE增强局部纹理
# =========================
clahe = cv2.createCLAHE(
    clipLimit=3.0,
    tileGridSize=(8,8)
)

img_clahe = clahe.apply(img_vis)



# =========================
# 显示
# =========================
plt.figure(figsize=(12,4))


plt.subplot(1,3,1)
plt.imshow(img, cmap="gray")
plt.title("Original High Frequency")
plt.axis("off")


plt.subplot(1,3,2)
plt.imshow(img_vis, cmap="gray")
plt.title("Amplified High Frequency")
plt.axis("off")


plt.subplot(1,3,3)
plt.imshow(img_clahe, cmap="turbo")
plt.title("CLAHE + Colormap")
plt.axis("off")


plt.tight_layout()
plt.show()



# =========================
# 保存结果
# =========================

cv2.imwrite(
    "high_frequency_visualization.png",
    img_clahe
)

print("saved: high_frequency_visualization.png")