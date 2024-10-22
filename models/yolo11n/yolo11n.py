import time

import cv2
from ultralytics import YOLO

from modules.tools import (
    check_file_extension,
    get_path_from_filename,
    get_video_info,
    open_data_json,
    save_data_json,
    set_output_path,
)

modelname = "yolo11n-pose"
model = YOLO(f"models\\yolo11n\\model\\{modelname}.pt")


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


def set_inference_args(vid_size):
    return {
        "conf": 0.7,  # confidence threshold
        "iou": 0.5,  # NMS IoU threshold
        "max_det": 1,  # maximum number of detections per image
        "stream": True,  # stream results
        # "show-boxes": False,  # display bounding boxes
        "imgsz": vid_size,  # inference size (pixels)
    }


def process_video(filename):
    """Process video frame by frame and save the output."""
    input_path = get_path_from_filename(filename)
    output_path = set_output_path(modelname, filename)

    vid = cv2.VideoCapture(input_path)
    width, height, fps, frames, duration = get_video_info(vid)

    vid_size = (width, height)
    inference_args = set_inference_args(vid_size)

    # Ustawienia dla zapisu wideo
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Przetwarzanie wideo klatka po klatce
    results = model(input_path, device="cpu", **inference_args)

    execution_time = 0
    for result in results:
        start_time = time.time()
        result = frame_visualization(result)
        frame = result.plot()  # plot the results on the frame
        out.write(frame)  # Zapisz klatkę do pliku wideo
        end_time = time.time()
        execution_time += end_time - start_time

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

    elif file_type == "image":
        results, process_time, frames = process_image(filename)
        avg_frame_time = process_time / frames
        execution_time = time.time() - start_time

    else:
        print("Unsupported file type")
        execution_time = time.time() - start_time
        return "Unsupported file type", 0, execution_time

    print_report(avg_frame_time, execution_time)
    performance_report(avg_frame_time, execution_time, filename)
    return results


def print_report(avg_frame_time, execution_time):
    """Print performance report."""
    print(
        f"""
        {modelname}:
        Average frame time:   {avg_frame_time: .5f} s,
        Total execution_time: {execution_time: .5f} s
        """
    )
    return avg_frame_time, execution_time


def performance_report(avg_frame_time, execution_time, filename: str):
    """Save performance report to data.json."""

    # Load existing data from data.json
    data = open_data_json()

    # Update the data with new performance metrics
    if modelname not in data["models"]:
        data["models"][modelname] = {}

    data["models"][modelname][filename] = {"avg_frame_time": avg_frame_time, "execution_time": execution_time}

    # Save updated data back to data.json
    save_data_json(data)
