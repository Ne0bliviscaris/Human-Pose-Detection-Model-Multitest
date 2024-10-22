import time

import models.yolo11.yolo11 as yolo11

vid = "vid1_400.mp4"
img = "1.jpg"

results, avg_frame_time, execution_time = yolo11.process_file(vid)
print(
    f"""Average frame time:{avg_frame_time: .5f} s,
      Total execution_time: {execution_time: .5f} s"""
)
