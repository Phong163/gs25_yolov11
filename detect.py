import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
from tracker.tracker import BYTETracker
import torch
from ultralytics.utils.metrics import box_iou

# Đường dẫn đến file video
video_path = r"C:\Users\OS\Desktop\gs25\2.mp4"  # Thay bằng đường dẫn đến file video của bạn

# Khởi tạo video capture
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Lỗi: Không thể mở file video {video_path}")
    exit()

# Khởi tạo tracker
tracker = BYTETracker(track_thresh=0.5, match_thresh=0.7, track_buffer=200, frame_rate=7)

# Khởi tạo mô hình YOLOv11
model_path = r"C:\Users\OS\Desktop\gs25\weights\last_yolov11s_15_7.onnx"
try:
    model = YOLO(model_path, task="detect")
    print("Đã tải mô hình YOLOv11 thành công với ONNX Runtime")
except Exception as e:
    print(f"Lỗi khi tải mô hình YOLO: {e}")
    cap.release()
    exit()

# Cơ sở dữ liệu đặc trưng và thời gian cho ReID
database_features = {} 
database_times = {}     
frame_last_seen = {}    
person_counter = 1    
frame_count = 0         

# Dictionary để lưu đặc trưng và nhãn cho từng track_id
track_features = {}     # {track_id: (feature, matched_id)}
track_to_person = {}    # {track_id: person_id}

def get_zone_coords(frame, zone_relative):
    height, width = frame.shape[:2]
    points_abs = [(int(x * width), int(y * height)) for x, y in zone_relative]
    return np.array([points_abs], dtype=np.int32)


def is_box_in_zone(box, zone_coords, score):
    x1, y1, x2, y2 = box
    box_area = [(x1, y1, x2, y2)]
    zone_points = zone_coords[0]
    zone_x = [point[0] for point in zone_points]
    zone_y = [point[1] for point in zone_points]
    zone_x1, zone_y1 = min(zone_x), min(zone_y)
    zone_x2, zone_y2 = max(zone_x), max(zone_y)
    zone_area = [(zone_x1, zone_y1, zone_x2, zone_y2)]
    box_xyxy = np.array(box_area)
    zone_xyxy = np.array(zone_area)
    iou = box_iou(torch.tensor(box_xyxy), torch.tensor(zone_xyxy)).item()
    return iou > score

def rescale(frame, img_size, x_min, y_min, x_max, y_max):
    scale_x = frame.shape[1] / img_size
    scale_y = frame.shape[0] / img_size
    return x_min * scale_x, y_min * scale_y, x_max * scale_x, y_max * scale_y
# Định nghĩa các vùng zone
zone_1 = [
    (0.6375, 0.5125),  # Điểm 1
    (0.4583, 0.4895),  # Điểm 2
    (0.1166, 0.9937),  # Điểm 3
    (0.4583, 0.9875)   # Điểm 4
]
zone_2 = [
    (0.7021, 0.8125),  # Điểm 1
    (0.5417, 0.7812),  # Điểm 2
    (0.4688, 0.9896),  # Điểm 3
    (0.6667, 0.9917)   # Điểm 4
]
zone_3 = [
    (0.7542, 0.5583),  # Điểm 1
    (0.8667, 0.5708),  # Điểm 2
    (0.8771, 0.9917),  # Điểm 3
    (0.7104, 0.9896)   # Điểm 4
]
zone_4 = [
    (0.6416, 0.5104),  # Điểm 1
    (0.5916, 0.7041),  # Điểm 2
    (0.6833, 0.7166),  # Điểm 3
    (0.6895, 0.6)
]
size = 480
prev_time = time.time()
track_person = {}
min_dist = 0.3
current_frame = 0
total_time_2 = 0
stopped_2 = True
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Đã hết video hoặc lỗi khi đọc frame")
        break
    h, w, _ = frame.shape
    size_i = (w,h)
    frame_count += 1  # Tăng số frame đã xử lý
    frame2 = cv2.resize(frame, (size, size))
    annotated_frame = frame.copy()

    # Tính tọa độ zone
    zone_coords_1 = get_zone_coords(annotated_frame, zone_1)
    zone_coords_2 = get_zone_coords(annotated_frame, zone_2)
    zone_coords_3 = get_zone_coords(annotated_frame, zone_3)
    zone_coords_4 = get_zone_coords(annotated_frame, zone_4)
    line_1 = (int(0.691*w), int(0.5529*h), int(0.5*w), int(0.988*h))
    # Chạy suy luận YOLOv11
   
    results = model.predict(frame2, stream=True, conf=0.6, iou=0.6, verbose=False)
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x_min_rescaled, y_min_rescaled, x_max_rescaled, y_max_rescaled = rescale(
                annotated_frame, size, x1, y1, x2, y2
            )
            conf = box.conf[0]
            cls = int(box.cls[0])
            detections.append([x_min_rescaled, y_min_rescaled, x_max_rescaled, y_max_rescaled, conf, cls])

    detections = np.array(detections) if detections else np.empty((0, 6))
    tracks = tracker.update(detections)
    current_frame += 1

    for track in tracks:
        tlbr = track.tlbr
        track_id = track.track_id
        box = (int(tlbr[0]), int(tlbr[1]), int(tlbr[2]), int(tlbr[3]))
        
        # Cập nhật last_seen_frame mỗi khi track được phát hiện
        if track_id not in track_person:
            # Khởi tạo thông tin người mới
            track_person[track_id] = {
                "start_time_1": time.time(),
                "start_time_2": time.time(),
                "stopped_1": False,
                "stopped_2": True,
                "total_time_1": 0,
                "total_time_2": 0,
                "last_seen_frame": current_frame,  # Chỉ giữ trường này
            }
        else:
            track_person[track_id]["last_seen_frame"] = current_frame

        if is_box_in_zone(box, zone_coords_1, 0.02) and not is_box_in_zone(box, zone_coords_3, 0.02):
            # Time_1
            if is_box_in_zone(box, zone_coords_2, 0.07):
                track_person[track_id]["stopped_1"] = True

            if not track_person[track_id]["stopped_1"]:
                track_person[track_id]["total_time_1"] = time.time() - track_person[track_id]["start_time_1"]

            if not track_person[track_id]["stopped_2"]:
                if track_person[track_id]["total_time_2"] == 0:
                    track_person[track_id]["start_time_2"] = time.time()
                track_person[track_id]["total_time_2"] = time.time() - track_person[track_id]["start_time_2"]
            
            if track_person[track_id]["stopped_1"]:
                track_person[track_id]["stopped_2"] = stopped_2

            # Vẽ thông tin lên khung hình
            cv2.putText(annotated_frame,
                        f"Person_ID: {track_id}, Time_1: {track_person[track_id]['total_time_1']:.1f}s, Time_2: {track_person[track_id]['total_time_2']:.1f}s",
                        (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2)

        # Time_2
        if is_box_in_zone(box, zone_coords_3, 0.1):
            if is_box_in_zone(box, zone_coords_4, 0.07):
                stopped_2 = False

        cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

    # Xóa các track không xuất hiện sau 500 frame
    to_delete = []
    for t_id, info in track_person.items():
        if current_frame - info["last_seen_frame"] > 200:
            to_delete.append(t_id)

    for t_id in to_delete:
        print(f"Deleted track_id {t_id}")
        del track_person[t_id]

    # Vẽ zone
    cv2.polylines(annotated_frame, zone_coords_1, isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.polylines(annotated_frame, zone_coords_2, isClosed=True, color=(0, 255, 255), thickness=2)
    cv2.polylines(annotated_frame, zone_coords_3, isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.polylines(annotated_frame, zone_coords_4, isClosed=True, color=(255, 0, 0), thickness=2)
    # Hiển thị FPS
    annotated_frame = cv2.resize(annotated_frame, (720, 640))
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if current_time > prev_time else 1.0
    prev_time = current_time
    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    cv2.imshow("Phát hiện YOLOv11 Video", annotated_frame)

    # except Exception as e:
    #     print(f"Lỗi trong quá trình suy luận: {e}")
    #     continue

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Dọn dẹp
track_features.clear()
track_to_person.clear()
frame_last_seen.clear()
cap.release()
cv2.destroyAllWindows()