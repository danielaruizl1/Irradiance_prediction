import os
import cv2
import time
import numpy as np
import humanfriendly
import matplotlib.pyplot as plt
from tqdm import tqdm

def create_video_from_frames(input_directory, output_video_path, ext, fps=5, colormap=False):
    
    # Get the list of files in the input directory
    file_list = os.listdir(input_directory)
    image_files = [f for f in file_list if f.endswith(ext)]

    # Get the first image to get dimensions for the video
    first_image_path = os.path.join(input_directory, image_files[0])
    first_image = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)
    first_image = cv2.cvtColor(first_image, cv2.COLOR_GRAY2BGR)
    height, width, channels = first_image.shape

    # Create the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For MP4 format, you can change this based on your needs
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write each frame to the video
    for image_file in tqdm(image_files):
        breakpoint()
        image_path = os.path.join(input_directory, image_file)
        frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if colormap:
            frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
            os.makedirs(os.path.join("colormaps", input_directory), exist_ok=True)
            cv2.imwrite(os.path.join("colormaps", input_directory, image_file), frame)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            min_val = frame.min()
            max_val = frame.max()
            frame = ((frame - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        video_writer.write(frame)

    # Release the video writer and close the video file
    video_writer.release()
    cv2.destroyAllWindows()

start_time = time.time()

input_directory = '20230822'

#create_video_from_frames(input_directory, f'Videos/{input_directory}_bw.mp4', ext='.png')
create_video_from_frames(input_directory, f'Videos/{input_directory}_colormap.mp4', colormap=True, ext='.png')
#create_video_from_frames(input_directory, f'Videos/{input_directory}_jp2.mp4', colormap=True, ext='.jp2')

elapsed = time.time() - start_time
print('Finished converting to video in {}'.format(humanfriendly.format_timespan(elapsed)))
