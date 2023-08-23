# 2023/08/02 14:00 Tokol1
# yolov5と同じフォルダにmain8202.pyを置く


import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

# YOLOv5モデルの読み込み(Load My learned YOLOv5 model)
model = torch.hub.load("yolov5", 'custom', path='yolov5_banana_streamlit//best1.pt',source='local')

# Streamlitアプリ名(Streamlit app name)
st.title("YOLOv5 Object Detection with Streamlitああ")

# ウェブカメラを開く(Open webcam)
# cap = cv2.VideoCapture(0)の場合は、PCに接続されたカメラを使用します。
# If cap = cv2.VideoCapture(0), use the camera connected to the PC.
# cap = cv2.VideoCapture(1)の場合は、PCに接続されたカメラのうち、2番目に接続されたカメラを使用します。
# If cap = cv2.VideoCapture(1), use the second camera connected to the PC.
cap = cv2.VideoCapture(0)

# 画像表示のための最大幅を設定(Set maximum width for displaying images)
max_width = 800

# 画像表示用のプレースホルダーを作成(Create a placeholder for the image)
placeholder = st.empty()

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # BGR画像をRGBに変換(Convert BGR image to RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 推論を実行(Perform inference)
    image = Image.fromarray(frame_rgb)
    results = model(image)

    # 検出結果を表示(Display detection results)
    # 必要なし(Not needed)
    #st.write("### Detection Results:")
    #st.dataframe(results.pandas().xyxy[0])

    # フレームに境界ボックスを描画(Draw bounding boxes on the frame)
    annotated_frame = np.array(results.render()[0])

    # 境界ボックス付きのリアルタイム画像を表示(Display image with bounding boxes)
    placeholder.image(annotated_frame, caption="Detection Result", use_column_width=True, channels="RGB", output_format="auto")

    # 'q'キーが押されたらループを終了(Break the loop when 'q' is pressed)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# キャプチャを解放(Release the capture)
cap.release()
cv2.destroyAllWindows()
