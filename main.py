import argparse
import cv2
import numpy as np
import pytz
import torch
from ultralytics import YOLO
import time
import threading
from tracker.tracker import BYTETracker
from utils import get_zone_coords, is_box_in_zone, rescale, extract_feature, cosine_similarity, setup_logger, send_time_to_kafka
from config.params import camera_configs
from rtsp_stream import RTSPStream
from torchreid.utils import FeatureExtractor
logger = setup_logger()
from datetime import datetime

class CustomerTracker:
    def __init__(self, camera_id, yolo_model_path, osnet_model_path, output_path, size=480, save_video=False, send_api=False):
        """Initialize the CustomerTracker with necessary components."""
        if camera_id not in camera_configs:
            raise ValueError(f"No configuration found for camera_id {camera_id}")
        
        # Load camera-specific configuration
        config = camera_configs[camera_id]
        self.rtsp_url = config["rtsp_url"]
        self.camera_id = camera_id
        self.zone_1 = config["zone_1"]
        self.zone_2 = config["zone_2"]
        self.zone_3 = config["zone_3"]
        self.zone_4 = config["zone_4"]
        self.zone_5 = config["zone_5"]
        self.track_person = config["track_person"]
        self.current_frame = config["current_frame"]
        self.yolo_model_path = yolo_model_path
        self.osnet_model_path = osnet_model_path
        self.output_path = output_path
        self.size = size
        self.save_video = save_video
        self.send_api = send_api
        self.track_old_customer = []

        # Initialize RTSP stream
        self.rtsp_stream = RTSPStream(self.rtsp_url, self.camera_id)
        self.rtsp_stream.start()

        # Initialize tracker
        self.tracker = BYTETracker(track_thresh=0.6, match_thresh=0.7, track_buffer=60, frame_rate=15)

        # Initialize YOLO model
        try:
            self.model = YOLO(self.yolo_model_path, task="detect")
            logger.info(f"Successfully loaded YOLOv11 model for camera {camera_id}")
        except Exception as e:
            logger.error(f"Error loading YOLO model for camera {camera_id}: {e}")
            self.rtsp_stream.stop()
            raise

        # Initialize OSNet feature extractor
        self.extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path=self.osnet_model_path,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Get video information
        self.fps = int(self.rtsp_stream.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.rtsp_stream.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.rtsp_stream.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize VideoWriter if save_video is True
        if self.save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Append camera_id to output path to avoid overwriting
            output_file = output_path.replace('.mp4', f'_cam{camera_id}.mp4')
            self.out = cv2.VideoWriter(output_file, fourcc, self.fps, (self.width, self.height))

    def process_frame(self, frame):
        """Process a single frame and return annotated frame."""
        self.current_frame += 1
        frame_resize = cv2.resize(frame, (self.size, self.size))
        annotated_frame = frame.copy()

        # Get zone coordinates
        zone_coords_1 = get_zone_coords(annotated_frame, self.zone_1)
        zone_coords_2 = get_zone_coords(annotated_frame, self.zone_2)
        zone_coords_3 = get_zone_coords(annotated_frame, self.zone_3)
        zone_coords_4 = get_zone_coords(annotated_frame, self.zone_4)
        zone_coords_5 = get_zone_coords(annotated_frame, self.zone_5)

        # Perform detection
        results = self.model.predict(frame_resize, stream=True, conf=0.6, iou=0.6, verbose=False)
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu())  # Chuyển về CPU
                x_min_rescaled, y_min_rescaled, x_max_rescaled, y_max_rescaled = rescale(
                    annotated_frame, self.size, x1, y1, x2, y2
                )
                conf = box.conf[0].cpu().item()  # Chuyển về CPU
                cls = int(box.cls[0].cpu())
                detections.append([x_min_rescaled, y_min_rescaled, x_max_rescaled, y_max_rescaled, conf, cls])

        detections = np.array(detections) if detections else np.empty((0, 6))
        tracks = self.tracker.update(detections)
        track_customer = []
        track_staff = []
        stopped_2 = True

        # Process tracks
        for track in tracks:
            tlbr = track.tlbr
            track_id = track.track_id
            box = (int(tlbr[0]), int(tlbr[1]), int(tlbr[2]), int(tlbr[3]))
            
            if is_box_in_zone(box, zone_coords_1, 0.02) and not is_box_in_zone(box, zone_coords_3, 0.02):
                track_customer.append(track_id)
            if is_box_in_zone(box, zone_coords_3, 0.1):
                track_staff.append(track_id)
                if is_box_in_zone(box, zone_coords_4, 0.07):
                    stopped_2 = False

        # Process customer tracks
        for track in tracks:
            tlbr = track.tlbr
            track_id = track.track_id
            box = (int(tlbr[0]), int(tlbr[1]), int(tlbr[2]), int(tlbr[3]))

            if track_id in track_customer:
                if track_id not in self.track_person:
                    feature, crop = extract_feature(self.extractor, frame, box)
                    if feature is None:
                        continue
                    matched_id = None
                    for track_id_person, info in self.track_person.items():
                        if track_id_person not in self.track_old_customer and "feature" in info:
                            sim = cosine_similarity(feature, info["feature"])
                            if sim > 0.7:
                                matched_id = track_id_person

                    if matched_id is not None:
                        self.track_person[track_id] = self.track_person[matched_id].copy()
                        self.track_person[track_id]["last_seen_frame"] = self.current_frame
                        self.track_person[track_id]["feature"] = feature
                        self.track_person[matched_id]["reid"] = True
                    else:
                        date_time = datetime.now(pytz.timezone("Asia/Ho_Chi_Minh")).strftime("%d-%m-%Y_%H-%M")
                        self.track_person[track_id] = {
                            "name_track_id": f"{track_id}_{date_time}_{self.camera_id}_{int(tlbr[0])}-{int(tlbr[1])}",
                            "start_time_1": time.time(),
                            "start_time_2": time.time(),
                            "stopped_1": False,
                            "stopped_2": True,
                            "total_time_1": 0,
                            "total_time_2": 0,
                            "last_seen_frame": self.current_frame,
                            "feature": feature,
                            "reid" : False,
                        }
                else:
                    self.track_person[track_id]["last_seen_frame"] = self.current_frame

                # Update timing
                if not self.track_person[track_id]["stopped_1"]:
                    self.track_person[track_id]["total_time_1"] = time.time() - self.track_person[track_id]["start_time_1"]
                if is_box_in_zone(box, zone_coords_2, 0.07) or (self.zone_5 and is_box_in_zone(box, zone_coords_5, 0.03)):
                    self.track_person[track_id]["stopped_1"] = True

                if not self.track_person[track_id]["stopped_2"]:
                    if self.track_person[track_id]["total_time_2"] == 0:
                        self.track_person[track_id]["start_time_2"] = time.time()
                    self.track_person[track_id]["total_time_2"] = time.time() - self.track_person[track_id]["start_time_2"]

                if self.track_person[track_id]["stopped_1"] and self.track_person[track_id]["stopped_2"]:
                    
                    self.track_person[track_id]["stopped_2"] = stopped_2

                if self.save_video:
                    # Draw tracking information
                    cv2.putText(
                        annotated_frame,
                        f" Time_1: {self.track_person[track_id]['total_time_1']:.1f}s, Time_2: {self.track_person[track_id]['total_time_2']:.1f}s",
                        (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 0),
                        3
                    )
            if self.save_video:
                cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

        self.track_old_customer = track_customer

        # Remove old tracks
        to_remove = []
        for t_id, info in self.track_person.items():
            if self.current_frame - info["last_seen_frame"] > 100:
                if info["stopped_1"] and not info["reid"] and self.send_api:
                    send_time_to_kafka(1, self.camera_id, info["name_track_id"], info["total_time_1"])
                if info["total_time_2"] > 0 and not info["reid"] and self.send_api:
                    send_time_to_kafka(2, self.camera_id, info["name_track_id"], info["total_time_2"])
                to_remove.append(t_id)

        # Xóa các track sau khi lặp
        for t_id in to_remove:
            del self.track_person[t_id]

        # Draw zone
        if self.save_video:
            cv2.polylines(annotated_frame, zone_coords_1, isClosed=True, color=(0, 0, 255), thickness=2)
        return annotated_frame

    def run(self):
        """Main loop to process RTSP stream."""
        while self.rtsp_stream.running:
            ret, frame = self.rtsp_stream.get_frame()
            if not ret or frame is None:
                time.sleep(0.2)
                continue

            annotated_frame = self.process_frame(frame)
            if self.save_video:
                self.out.write(annotated_frame)
                cv2.imshow(f"Phát hiện YOLOv11 RTSP - Camera {self.camera_id}", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        self.rtsp_stream.stop()
        # Cleanup
        if self.save_video and hasattr(self, 'out'):
            self.out.release()
            cv2.destroyAllWindows()

    def __del__(self):
        """Cleanup resources when object is destroyed."""
        if self.save_video and hasattr(self, 'out'):
            self.out.release()
        if hasattr(self, 'rtsp_stream'):
            self.rtsp_stream.stop()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Counting the waiting time of customer with RTSP')
    parser.add_argument('--camera_id', type=int, nargs='*', default=None,
                        help='Camera ID(s) to process (default: all cameras in camera_configs)')
    parser.add_argument('--yolo_model', type=str, default='./weights/best_yolov11_4k.onnx',
                        help='Path to YOLO model')
    parser.add_argument('--osnet_model', type=str, default='./weights/osnet_x1_0_imagenet.pth',
                        help='Path to OSNet model')
    parser.add_argument('--output', type=str, default='./output/output_video.mp4',
                        help='Base path to output video (camera_id will be appended)')
    parser.add_argument('--imgsz', type=int, default=480,
                        help='Size for resizing frames')
    parser.add_argument('--save_video', action='store_true',
                        help='Whether to save the output video')
    parser.add_argument('--send_api', action='store_true',
                        help='Whether to save the output video')
    return parser.parse_args()

def run_tracker_for_camera(camera_id, yolo_model_path, osnet_model_path, output_path, size, save_video, send_api):
    """Run a CustomerTracker for a specific camera in a separate thread."""
    try:
        tracker = CustomerTracker(
            camera_id=camera_id,
            yolo_model_path=yolo_model_path,
            osnet_model_path=osnet_model_path,
            output_path=output_path,
            size=size,
            save_video=save_video,
            send_api = send_api
        )
        tracker.run()
    except Exception as e:
        logger.error(f"Error running tracker for camera {camera_id}: {e}")

if __name__ == "__main__":
    args = parse_args()
    
    # If no camera_id is specified, run all cameras in camera_configs
    camera_ids = args.camera_id if args.camera_id is not None else list(camera_configs.keys())
    
    # Create and start a thread for each camera
    threads = []
    for camera_id in camera_ids:
        thread = threading.Thread(
            target=run_tracker_for_camera,
            args=(camera_id, args.yolo_model, args.osnet_model, args.output, args.imgsz, args.save_video, args.send_api)
        )
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()
