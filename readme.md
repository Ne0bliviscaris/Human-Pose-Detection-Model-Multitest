# IDEA

App for testing multiple 3D pose detection models with one click

- Inputs folder
    - contains test videos

- Outputs folder
    - contains folders named after each model
        - processed videos are moved to 'processed' subfolder
    - each folder contains processed videos

- CSV table
    - Measures time used for each video to process

- Archive
    - check if file name is already in 'processed'
        - prompt if the file is not already processed
        - if it is, show the file and processed outputs


### Future possibilities
- Add 3d environment render
- Add Streamlit interface
    - Show videos in real time
    - Show time it took to process video
    - Video details (size, resolution)
    - Show graphs
        - time used per model
        - time to process all images for model