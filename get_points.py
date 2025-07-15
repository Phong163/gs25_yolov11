import cv2
import numpy as np

# Danh sách để lưu tọa độ các điểm
points = []

# Hàm xử lý sự kiện chuột
def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:  # Nhấp chuột trái để lấy tọa độ
        points.append((x, y))
        print(f"Điểm {len(points)}: ({x}, {y})")
        # Vẽ vòng tròn tại điểm nhấp chuột
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Image", image)
        # Khi có 4 điểm, tự động dừng
        if len(points) == 4:
            print("Đã lấy đủ 4 điểm:", points)
            cv2.destroyAllWindow()

# Đọc ảnh
image = cv2.imread(r"C:\Users\OS\Desktop\gs25\images\no7_0001.jpg")  # Thay "path_to_your_image.jpg" bằng đường dẫn ảnh của bạn
image = cv2.resize(image, (480,480))
if image is None:
    print("Không thể tải ảnh!")
    exit()

# Hiển thị ảnh
cv2.imshow("Image", image)

# Thiết lập sự kiện chuột
cv2.setMouseCallback("Image", mouse_callback)

# Chờ người dùng nhấn 'q' để thoát hoặc tự động thoát khi có 4 điểm
while True:
    if cv2.waitKey(1) & 0xFF == ord('q') or len(points) == 4:
        break

# Hiển thị tọa độ cuối cùng
print("Tọa độ 4 điểm:", points)

# Giải phóng cửa sổ
cv2.destroyAllWindows()