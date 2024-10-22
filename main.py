from modules.multi_model import run_all_models
from modules.tools import get_all_filenames_in_inputs

files = get_all_filenames_in_inputs()

# file = files[2]
for file in files:
    run_all_models(file)
