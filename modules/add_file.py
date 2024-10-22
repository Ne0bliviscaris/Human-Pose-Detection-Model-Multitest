import json
import os
import shutil

import cv2

from modules.tools import check_file_extension, get_video_info


def add_file(file_path: str):
    """Add file to inputs folder and update data.json with file information."""

    filename, destination_path = copy_file_to_inputs(file_path)

    # Check file type
    file_type = check_file_extension(filename)

    # Get file information
    file_info = get_file_info(destination_path, file_type)
    update_json(file_type, filename, file_info)

    return destination_path


def get_file_info(file_path: str, file_type: str):
    """Process file information and update data.json."""

    # Get file size and name
    file_info = {
        "file_size": f"{os.path.getsize(file_path) / (1024 * 1024):.2f}MB",
        "file": os.path.basename(file_path),
    }

    if file_type == "video":
        add_video_info(file_path, file_info)
    elif file_type == "image":
        add_img_info(file_path, file_info)

    return file_info


def add_video_info(video, file_info):
    vid = cv2.VideoCapture(video)
    width, height, fps, frames = get_video_info(vid)
    file_info.update({"video_height": height, "video_width": width, "fps": fps, "frames": frames})
    vid.release()


def add_img_info(img, file_info):
    img = cv2.imread(img)
    height, width, _ = img.shape
    file_info.update({"height": height, "width": width})


def update_json(file_type, filename, file_info):
    # Load data.json
    data_file = "modules\\data.json"

    with open(data_file, "r") as file:
        data = json.load(file)

    # Update data.json
    if file_type == "video":
        if "videos" not in data:
            # Add missing category
            data["videos"] = {}
        data["videos"][filename] = file_info
    elif file_type == "image":
        if "images" not in data:
            # Add missing category
            data["images"] = {}
        data["images"][filename] = file_info

    # Save changes to data.json
    with open(data_file, "w") as file:
        json.dump(data, file, indent=4)


def copy_file_to_inputs(file_path):
    # Copy file to inputs folder
    inputs = "inputs"
    os.makedirs(inputs, exist_ok=True)
    filename = os.path.basename(file_path)
    destination_path = os.path.join(inputs, filename)
    if not file_path == destination_path:
        shutil.copy(file_path, destination_path)
    else:
        print("File already exists in inputs folder.")
    return filename, destination_path
