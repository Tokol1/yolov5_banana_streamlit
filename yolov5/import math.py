import math

import torch
import cv2

# モデルの読み込み
model = torch.hub.load(".", "custom", path="runs/train/exp2/weights/best.pt", source="local")

# 入力画像の読み込み
img = cv2.imread("input.jpg")

# 検出の閾値設定
model.conf = 0.5

# 物体検出
result = model(img)

# バウンディングボックスを取得し画像をクリップ
for idx, row in enumerate(result.pandas().xyxyn[0].itertuples()):
    height, width = img.shape[:2]

    xmin = math.floor(width * row.xmin)
    xmax = math.floor(width * row.xmax)
    ymin = math.floor(height * row.ymin)
    ymax = math.floor(height * row.ymax)

    cripped_img = img[ymin:ymax, xmin:xmax]

    cv2.imwrite(f"crip_{idx}.jpg", cripped_img)
