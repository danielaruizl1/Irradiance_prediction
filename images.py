import matplotlib.pyplot as plt
import glob

# List all the JP2 images in the folder
list_paths = glob.glob('20230131\*.jp2')
print(len(list_paths))

# Create a figure and subplots in a 3x2 grid
fig, axs = plt.subplots(2, 3, figsize=(12, 8))

# Read the images 
image_1 = plt.imread(list_paths[-6])
image_2 = plt.imread(list_paths[-5])
image_3 = plt.imread(list_paths[-4])
image_4 = plt.imread(list_paths[-3])
image_5 = plt.imread(list_paths[-2])
image_6 = plt.imread(list_paths[-1])

# Show the images in the subplots
axs[0, 0].imshow(image_1, cmap='gray')
axs[0, 0].axis('off')

axs[0, 1].imshow(image_2, cmap='gray')
axs[0, 1].axis('off')

axs[0, 2].imshow(image_3, cmap='gray')
axs[0, 2].axis('off')

axs[1, 0].imshow(image_4, cmap='gray')
axs[1, 0].axis('off')

axs[1, 1].imshow(image_5, cmap='gray')
axs[1, 1].axis('off')

axs[1, 2].imshow(image_6, cmap='gray')
axs[1, 2].axis('off')

# Adjust the spacing between subplots to avoid overlapping
plt.tight_layout()

# Show the subplot grid
plt.show()

