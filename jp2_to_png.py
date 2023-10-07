#%%
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def get_max_and_min(input_directory):
    # Get the list of jp2 files in the input directory
    file_list = os.listdir(input_directory)
    image_files = [f for f in file_list if f.endswith(".jp2")]
    min_global = np.inf
    max_global = 0
    
    for image_file in tqdm(image_files):
        image_path = os.path.join(input_directory, image_file)
        jp2_img = plt.imread(image_path)
        min_img = np.min(jp2_img)
        max_img = np.max(jp2_img)

        if min_img < min_global:
            min_global = min_img
        if max_img > max_global:
            max_global = max_img

    return {"min":min_global, "max":max_global}

input_directory = '20230822'
norm_dict = get_max_and_min(input_directory)
print(norm_dict)
#%%
norm_dict = {'min': 13438, 'max': 43114}

def convert_jp2_to_png(input_directory, ext, cmap):

    # Get the list of jp2 files in the input directory
    file_list = os.listdir('20230822-color')
    image_files = [f for f in file_list if f.endswith(ext)]
    image_files = [f.replace("png","jp2") for f in image_files]
    
    # Convert and save each jp2 image
    for image_file in tqdm(image_files):
        image_path = os.path.join(input_directory, image_file)
        plt.figure()
        jp2_img = plt.imread(image_path)
        jp2_img = ((jp2_img - norm_dict["min"])/(norm_dict["max"]-norm_dict["min"]))*255
        plt.imshow(jp2_img, cmap=cmap)
        plt.axis("off")
        plt.savefig(input_directory+"/"+image_file[:-3]+"png", bbox_inches='tight', pad_inches=0)
        plt.close()

convert_jp2_to_png(input_directory, ".png", "gray")