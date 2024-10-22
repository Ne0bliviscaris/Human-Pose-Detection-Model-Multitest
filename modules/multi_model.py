import models.yolo11n.yolo11n as yolo11n


def run_all_models(file):
    """Run all models for given file and measure performance"""
    yolo11n.process_file(file)
