import subprocess

def cut_video(input_file, output_file, start_time, end_time):
    command = [
        "ffmpeg", "-i", input_file, "-ss", start_time, "-to", end_time, "-c", "copy", output_file
    ]
    subprocess.run(command, check=True)

# Ví dụ cắt video từ 10s đến 20s
cut_video(r"C:\Users\OS\Desktop\gs25\video\vlc-record-2025-07-07-16h28m19s-rtsp___115.78.133.22_554_Streaming_Channels_201-.mp4", "2.mp4", "00:11:37", "00:17:00")
