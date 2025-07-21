import datetime
from venv import logger
import numpy as np
import pytz
import requests
import torch
from ultralytics.utils.metrics import box_iou


import logging
import logging.handlers
logger = logging.getLogger('metric_8_9')

def setup_logger(
    logger_name='metric_8_9_gs25',
    log_file='output/metric_8_9_gs25.log',
    level=logging.DEBUG,
    max_bytes=10*1024*1024,
    backup_count=5,
    log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
):
    """
    Thiết lập logger với console và file handler (rotation).
    
    Args:
        logger_name (str): Tên của logger.
        log_file (str): Đường dẫn file log.
        level (int): Mức độ log.
        max_bytes (int): Kích thước tối đa của file log.
        backup_count (int): Số file backup tối đa.
        log_format (str): Định dạng log.
    
    Returns:
        logging.Logger: Đối tượng logger đã được cấu hình.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False
    
    formatter = logging.Formatter(log_format)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

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

def send_number_to_backend(last_plate_number, time, base_url="http://10.0.1.22:8000"):
    data = {
        "box_id": "214234352",
        "customer": "17_7_2025_05_19_34236534",
        "checkout_to_exit_time": time, # time dạng float tính theo giây (12.4s)
    }
    url = f"{base_url}/api/v1/receive-number-of-hanger"
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(url, json=data, headers=headers)
        if response.status_code in (200, 201):
            try:
                result = response.json()
                logger.info(f"Thành công gửi số: {last_plate_number}, Timestamp: {time}")
            except ValueError:
                logger.info(f"Thành công gửi số: {last_plate_number}, Timestamp: {time}")
        else:
            logger.error(f"Lỗi API: {response.status_code}, {response.text}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Yêu cầu API thất bại: {str(e)}")
        
def extract_feature(extractor, image, box):
    x1, y1, x2, y2 = box
    crop = image[y1:y2, x1:x2]
    if crop.size > 0:
        feature = extractor(crop)
        return feature.cpu().numpy(),crop
    return None,None

def cosine_similarity(feat1, feat2):
    # Làm phẳng mảng nếu là 2D
    feat1 = feat1.flatten() if len(feat1.shape) > 1 else feat1
    feat2 = feat2.flatten() if len(feat2.shape) > 1 else feat2
    # Tính tích vô hướng và chuẩn
    dot_product = np.dot(feat1, feat2)
    norm_product = np.linalg.norm(feat1) * np.linalg.norm(feat2)
    # Tránh chia cho 0
    return dot_product / norm_product if norm_product != 0 else 0.0

