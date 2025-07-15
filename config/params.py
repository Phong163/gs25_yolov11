# Thiết lập tham số
img_size_ed = 640
img_size_sb = 480

red_line_ed = (380, 0, 380, 190)  # (x1, y1, x2, y2)
blue_line_ed = (550, 0, 550, 190)
red_line_sb = (1000, 0, 1000, 190)  # (x1, y1, x2, y2)
blue_line_sb = (850, 0, 850, 190) 

#Tham số nms 
conf_thres = 0.5 # Ngưỡng độ tin cậy
iou_thres = 0.4   # Ngưỡng IOU cho NMS
max_det = 1000    # Số lượng tối đa đối tượng phát hiện

max_frames = 1
max_number = 430
max_number_sb = 515

roi_sb = [700, 0, 600, 400]

# Define class names and colors for visualization
class_names = ['plate', 'number']
colors = [(0, 255, 0), (0, 0, 255)]  # Green for plate, Red for number

# Lưu trữ trạng thái nhận diện số cho từng track
track_plate = {}
track_text = {}  # {track_id: {"text": str, "recognized": bool, "stopped": bool, "frame_count": int, "cls": int}}
last_plate_number = 0  # Số của plate gần nhất
count = 0
current_number = 0  # Số hiện tại để gán cho plate_tracks


# Khởi tạo biến cho NumberFilter
N = []  # N0, N1, N2
accuracy = []    # Accuracy của N0, N1, N2
result = []             # Kết quả lưu trạng thái plates
state = 0               # Trạng thái hiện tại (0, 1, 2)
number_count = 0        # Đếm số plate có số
delete_flag = 0
set_stop = 0
delete_flag_max = False
haved_number_box = False
stop_ocr = False
set_stop_plate = False