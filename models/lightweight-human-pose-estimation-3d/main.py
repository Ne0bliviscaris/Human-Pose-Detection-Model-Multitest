import os

# Ustawienie ścieżki roboczej
os.chdir("models\\lightweight-human-pose-estimation-3d")
model = "human-pose-estimation-3d.pth"
model_dir = "modules\\lightweight-human-pose-estimation-3d"
model_path = os.path.join(model_dir, model)

INPUTS = "..\\..\\inputs"
OUTPUTS = "..\\..\\outputs"
