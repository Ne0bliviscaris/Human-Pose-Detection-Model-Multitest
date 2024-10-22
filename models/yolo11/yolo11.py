import time

import cv2
from ultralytics import YOLO

from modules.tools import (
    check_file_extension,
    get_path_from_filename,
    get_video_info,
    set_output_path,
)

modelname = "yolo11n-pose"
model = YOLO(f"models\\yolo11\\model\\{modelname}.pt")


def frame_visualization(frame):
    visualization_args = {
        "show_boxes": False,  # display bounding boxes
        "show_labels": False,  # display labels
        "show_conf": False,  # display confidence scores
    }
    if not visualization_args["show_boxes"]:
        frame.boxes = []  # Ukryj bounding boxy
    if not visualization_args["show_labels"]:
        frame.names = []  # Ukryj etykiety
    if not visualization_args["show_conf"]:
        frame.scores = []  # Ukryj pewności
    return frame


def process_video(filename):
    """Process video frame by frame and save the output."""
    input_path = get_path_from_filename(filename)
    output_path = set_output_path(modelname, filename)

    vid = cv2.VideoCapture(input_path)
    width, height, fps, frames = get_video_info(vid)

    # vid_size = (width, height)
    inference_args = {
        "conf": 0.7,  # confidence threshold
        "iou": 0.5,  # NMS IoU threshold
        "max_det": 1,  # maximum number of detections per image
        "stream": True,  # stream results
        # "show-boxes": False,  # display bounding boxes
        # "imgsz": vid_size,  # inference size (pixels)
    }

    # Ustawienia dla zapisu wideo
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Przetwarzanie wideo klatka po klatce
    results = model(input_path, device="cpu", **inference_args)

    start_time = time.time()
    for result in results:
        result = frame_visualization(result)
        frame = result.plot()  # plot the results on the frame
        out.write(frame)  # Zapisz klatkę do pliku wideo
    end_time = time.time()
    execution_time = end_time - start_time

    # Zwolnij zasoby
    vid.release()
    out.release()
    return results, execution_time, frames


def process_image(filename):
    """Process and save the output."""
    input_path = get_path_from_filename(filename)
    output_path = set_output_path(modelname, filename)

    inference_args = {
        "conf": 0.7,  # confidence threshold
        "iou": 0.5,  # NMS IoU threshold
        "max_det": 1,  # maximum number of detections per image
        "save": False,
        "show": False,
        # "stream": True,  # stream results
        # "imgsz": vid_size,  # inference size (pixels)
    }

    # Execute the model
    start_time = time.time()  # Start time measurement
    result = model(input_path, device="cpu", **inference_args)
    end_time = time.time()  # End time measurement

    run_time = end_time - start_time  # Total time taken
    frames = len(result)  # Number of frames processed

    # Save the output to a predefined path
    cv2.imwrite(output_path, result[0].plot())
    return result, run_time, frames


def process_file(filename):
    """Process the file based on its extension."""
    start_time = time.time()
    file_type = check_file_extension(filename)
    if file_type == "video":
        results, process_time, frames = process_video(filename)
        avg_frame_time = process_time / frames
        execution_time = time.time() - start_time
        return results, avg_frame_time, execution_time

    elif file_type == "image":
        results, process_time, frames = process_image(filename)
        execution_time = time.time() - start_time
        return results, process_time, execution_time

    else:
        print("Unsupported file type")
        execution_time = time.time() - start_time
        return "Unsupported file type", 0, execution_time


if __name__ == "__main__":
    # filename = "moj2_400.mp4"
    filename = "1.jpg"
    # process_video(filename)
    process_image(filename)
