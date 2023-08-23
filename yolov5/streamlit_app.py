import streamlit as st
import subprocess
import os
from PIL import Image

def main():
    st.title("YOLOv5 Object Detection with Streamlit")

    # ファイルのアップロード
    uploaded_file = st.file_uploader("画像または動画をアップロードしてください", type=["jpg", "jpeg", "png", "mp4"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image/Video.", use_column_width=True)

        # アップロードされたファイルでYOLOv5検出を実行
        if st.button("オブジェクト検出を実行"):
            # アップロードされたファイルを一時的な場所に保存
            temp_path = "/tmp"  # 一時ファイルのパスを適切に指定してください
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # detect.pyを実行するためのコマンドを構築
            cmd = [
                "python", "detect.py",
                "--weights", "yolov5m.pt",
                "--source", temp_path,
                # その他の引数をここに追加
                "--view-img",
                "--save-txt",
                "--exist-ok",
                "--classes", "47"
            ]

            # コマンドを実行
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # 出力画像を読み込んで表示
            output_image_path = "/temp/detected_image.jpg"  # detect.pyの出力画像のパスを指定
            if os.path.exists(output_image_path):
                image = Image.open(output_image_path)
                st.image(image, caption="Detected Objects", use_column_width=True)

            # 一時ファイルをクリーンアップ
            os.remove(temp_path)
            st.info("クリーンアップ完了")

if __name__ == "__main__":
    main()
