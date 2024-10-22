import os

import cv2


def get_video_info(vid):
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    return width, height, fps, frames


def get_path_from_filename(filename):
    return f"inputs\\{filename}"


def set_output_path(modelname, filename):
    """Set output path and ensure the directory exists."""
    output_path = f"outputs\\{modelname}\\{filename}"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    return output_path


def is_video_file(filename: str) -> bool:
    """Check if the file is a video based on its extension."""
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    _, ext = os.path.splitext(filename)
    return ext.lower() in video_extensions


def is_image_file(filename: str) -> bool:
    """Check if the file is an image based on its extension."""
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    _, ext = os.path.splitext(filename)
    return ext.lower() in image_extensions


def check_file_extension(filename: str):
    if is_image_file(filename):
        return "image"
    elif is_video_file(filename):
        return "video"
    else:
        return "unknown"
