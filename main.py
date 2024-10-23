from models.yolo11n import yolo11n
from modules.multi_model import run_all_models
from modules.tools import get_all_filenames_in_inputs

files = get_all_filenames_in_inputs()

file = files[0]
run_all_models(file)

# Cycle through all files in inputs directory
# for file in files:
#     run_all_models(file)

# yolo11n.process_file(file)
