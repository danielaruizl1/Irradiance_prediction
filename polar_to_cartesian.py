import cv2
import os
import numpy as np

def polar_to_cartesian(input_directory):

    file_list = os.listdir(input_directory)
    file_list = [f for f in file_list if f.endswith(".png")]

    for file in file_list:

        name=file[:-4]
    
        # Cargar la imagen del cielo 360 
        image_path = os.path.join(input_directory, file)
        sky360 = cv2.imread(image_path)

        # Obtener dimensiones de la imagen
        height, width, _ = sky360.shape

        # Convertir la imagen a coordenadas polares
        center = (width // 2, height // 2)
        max_radius = min(width, height) // 2
        polar_image = cv2.warpPolar(sky360, (max_radius, 360), center, max_radius, cv2.WARP_POLAR_LINEAR)
        polar_image = cv2.rotate(polar_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(os.path.join(input_directory,name+"_polar.jpg"), polar_image)


        """ # Dividir la imagen polar en cuatro partes (Norte, Este, Oeste, Sur)
        norte = polar_image[:, :90]
        este = polar_image[:, 90:180]
        sur = polar_image[:, 180:270]
        oeste = polar_image[:, 270:]


        # Guardar las imágenes divididas
        cv2.imwrite(f'{name}_norte.jpg', norte)
        cv2.imwrite(f'{name}este.jpg', este)
        cv2.imwrite(f'{name}sur.jpg', sur)
        cv2.imwrite(f'{name}oeste.jpg', oeste)"""

polar_to_cartesian(os.path.join("colormaps","20230822"))