import os

# Đường dẫn đến thư mục chứa ảnh
folder_path = r"C:\Users\OS\Desktop\thanglong\datasets_vpic\no5"  # Thay bằng đường dẫn thư mục của bạn

# Lấy danh sách tất cả các file ảnh trong thư mục
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif','.txt')  # Các định dạng ảnh hỗ trợ
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]

# Sắp xếp file để đảm bảo thứ tự nhất quán
image_files.sort()

# Đổi tên các file ảnh theo thứ tự tăng
for index, old_name in enumerate(image_files, start=1):
    # Lấy phần mở rộng của file
    file_extension = os.path.splitext(old_name)[1]
    # Tạo tên mới (ví dụ: 1.jpg, 2.jpg, ...)
    new_name = f"no5_{index}{file_extension}"
    # Đường dẫn đầy đủ của file cũ và mới
    old_file_path = os.path.join(folder_path, old_name)
    new_file_path = os.path.join(folder_path, new_name)
    
    # Đổi tên file
    os.rename(old_file_path, new_file_path)
    print(f"Đã đổi tên: {old_name} -> {new_name}")

print("Hoàn tất đổi tên các file ảnh!")