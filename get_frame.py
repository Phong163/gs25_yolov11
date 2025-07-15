import cv2
import os
import numpy as np

# Đường dẫn video và thư mục lưu frame
video_path = r'C:\Users\OS\Desktop\gs25\video\vlc-record-2025-07-08-15h34m12s-rtsp___115.78.133.22_554_Streaming_Channels_201-.mp4'
save_folder = r'C:\Users\OS\Desktop\gs25\images'
os.makedirs(save_folder, exist_ok=True)


# Load video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Không thể mở video!")
    exit()

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Kết thúc video

    # Lưu mỗi 5 frame
    if frame_count % 25 == 0:
        # Vẽ vùng cắt lên frame
        temp_frame = frame.copy()
        saved_count += 1
        save_path = os.path.join(save_folder, f"no8_{saved_count:04d}.jpg")
        cv2.imwrite(save_path, temp_frame)
        print(f"Đã lưu {save_path}")
        # if saved_count == 120:
        #     print('done')
        #     break

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Nhấn 'q' để thoát
            break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
print("Hoàn thành!")