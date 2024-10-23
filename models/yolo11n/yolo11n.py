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


def model_info():
    """Model information"""
    model_info = {
        "name": "yolo11n-pose",
        "description": "YOLOv5 model for pose estimation",
        "possible_inputs": "video, image, stream",
        "output_path": "outputs\\yolo11n-pose",
        "model_size": "5.96 MB",
        "model_path": "models\\yolo11n\\model\\yolo11n-pose.pt",
        "device": "cpu",
        "model_link": "https://docs.ultralytics.com/tasks/pose/",
    }
    return model_info


def initialize_model():
    """Initialize the yolo11n-pose model."""
    model = YOLO(f"models\\yolo11n\\model\\{modelname}.pt")
    return model


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
        # with stream mode can't extract keypoints
        "stream": False,  # stream results
        # "show-boxes": False,  # display bounding boxes
        "imgsz": vid_size,  # inference size (pixels)
    }


def process_video(model, filename):
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

    # print(f"\nFirst frame Keypoints:\n{results[0].keypoints}\n")  # Print keypoints from the first frame

    # Zwolnij zasoby
    vid.release()
    out.release()
    return results, execution_time, frames


def process_image(model, filename):
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

    # print(f"\nKeypoints:\n{result[0].keypoints}\n")  # Print keypoints

    # Save the output to a predefined path
    cv2.imwrite(output_path, result[0].plot())
    return result, run_time, frames


def process_file(filename: str):
    """Process the file based on its extension."""
    file_type = check_file_extension(filename)

    model_init_time = time.time()
    model = initialize_model()
    model_init_duration = time.time() - model_init_time

    processing_start_time = time.time()
    if file_type == "video":
        results, process_duration, frames = process_video(model, filename)
        avg_frame_time = process_duration / frames
        execution_time = time.time() - processing_start_time

    elif file_type == "image":
        results, process_duration, frames = process_image(model, filename)
        avg_frame_time = process_duration / frames
        execution_time = time.time() - processing_start_time

    else:
        print("Unsupported file type")
        execution_time = time.time() - processing_start_time
        return "Unsupported file type", 0, execution_time

    print_report(avg_frame_time, execution_time)
    save_performance_report(avg_frame_time, execution_time, model_init_duration, filename)
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


def save_performance_report(avg_frame_time, total_execution_time, model_init_duration, filename):
    """Save performance report to data.json."""

    # Load existing data from data.json
    data = open_data_json()
    data = ensure_json_structure(data)

    data["models"][modelname]["results"][filename] = {
        "avg_frame_time": f"{avg_frame_time: .5f}",
        "execution_time": f"{total_execution_time: .5f}",
        "file_path": f"outputs\\{modelname}\\{filename}",
        "model_init_time": f"{model_init_duration: .5f}",
    }

    # Save updated data back to data.json
    save_data_json(data)


def ensure_json_structure(data):
    """Ensure that the data.json structure is correct."""
    if "models" not in data:
        data["models"] = {}
    if modelname not in data["models"]:
        data["models"][modelname] = {}
        update_model_info()
    if "results" not in data["models"][modelname]:
        data["models"][modelname]["results"] = {}
    return data


def update_model_info():
    """Add model information to data.json."""
    data = open_data_json()
    if "models" not in data:
        data["models"] = {}
    if modelname not in data["models"]:
        data["models"][modelname] = {}
    if "results" not in data["models"][modelname]:
        data["models"][modelname]["results"] = {}

    data["models"][modelname] = model_info()
    save_data_json(data)
