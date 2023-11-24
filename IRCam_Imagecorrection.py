# -*- Image Barrel Correction*-
"""
Modified to its latest version on 24 November 2023

@author: Özge Sinem Özçakmak
ozgesinemozcakmak@gmail.com
Reference:
https://github.com/ozsioz/InfraDet-LTT

"""

import numpy as np
import math
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import ctypes
from PIL import Image,ImageEnhance
import PIL 
import os 
import cv2
import imageio
import imutils


import matplotlib
from skimage import transform

from skimage import data
from skimage import img_as_float

def main():


    global filename,path,fileout,counter,fint,inputfile
#Choosing all the images with .tiff extension in a folder by Asking for Directory
#Choose the ' RawImages ' file for a test case.

    ctypes.windll.user32.MessageBoxW('Image Location', 'Please select the folder where images are located', 0)
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askdirectory()
    counter = 0 # counter for analyzed number of images

    for filename in os.listdir(path):
        if filename.endswith('.tiff'): ##only works on tiff images
            counter += 1
            
            print (filename)
            imageSelect = os.path.join(path,filename)
            imagesel   = cv2.imread(imageSelect)
            
            if imagesel is None:
                print(f"Error: Unable to read image {imageSelect}")
                continue  # Skip to the next iteration of the loop)
       
           # h, w, z = get_image_height_and_width_and_channels(image)


## The matrix constants below should be arranged according to the camera location/distortion. It is same for all the images for a stationary camera during acquisition.

            paramA = -0  # affects only the outermost pixels of the image
            paramB =  0 # most cases only require b optimization
            paramC =  -0.08 # most uniform correction
            paramD = 1.0 - paramA - paramB - paramC  # describes the linear scaling of the image
 
            corrected_original_image = barrel_correction2(imagesel, paramA, paramB, paramC, paramD)

            figf,axf = plt.subplots()
            # make a plot
            axf.imshow(corrected_original_image)
            plt.show


            theta=8*np.pi/180
            tf_img= imutils.rotate(corrected_original_image, angle=-1)
    
     
            figf,axf = plt.subplots()
            axf.imshow(tf_img)
            axf.set_title(' transformation')
            plt.show
            imageio.imwrite(filename,tf_img)
            

 ## Saving the image with the same name
def save_image_array(image_array, fname):
    imageio.imwrite(fname, image_array)
 
# Checking the image dimensions
def get_image_height_and_width_and_channels(image):
    return image.shape[1], image.shape[0], image.shape[2]

 
# Barrel Distortion Correction
def barrel_correction2(src_image, param_a, param_b, param_c, param_d):
    xDim = get_image_height_and_width_and_channels(src_image)[0]
    yDim = get_image_height_and_width_and_channels(src_image)[1]
    zDim = get_image_height_and_width_and_channels(src_image)[2]
 
    dest_image = np.zeros_like(src_image)
 
    xcen = (xDim - 1.0) / 2.0
    ycen = (yDim - 1.0) / 2.0
    normDist = min(xcen, ycen)
 
    imageMin = np.min(src_image)
    dest_image.fill(imageMin)
 
    for k in range(zDim):
        for j in range(yDim):
            yoff = (j - ycen) / normDist
 
            for i in range(xDim):
                xoff = (i - xcen) / normDist
                rdest2 = xoff * xoff + yoff * yoff
                rdest = math.sqrt(rdest2)
                rdest3 = rdest2 * rdest
                rdest4 = rdest2 * rdest2
                rsrc = param_a * rdest4 + param_b * rdest3 + param_c * rdest2 + param_d * rdest
                rsrc = normDist * rsrc
                ang = math.atan2(yoff, xoff)
                xSrc = xcen + (rsrc * math.cos(ang))
                ySrc = ycen + (rsrc * math.sin(ang))
 
                if 0 <= xSrc < xDim - 1 and 0 <= ySrc < yDim - 1:
                    xBase = int(math.floor(xSrc))
                    delX = float(xSrc - xBase)
                    yBase = int(math.floor(ySrc))
                    delY = float(ySrc - yBase)
 
                    dest_image[j][i][k] = int((1 - delX) * (1 - delY) * src_image[yBase][xBase][k])
 
                    if xSrc < (xDim - 1):
                        dest_image[j][i][k] += int(delX * (1 - delY) * src_image[yBase][xBase + 1][k])
 
                    if ySrc < (yDim - 1):
                        dest_image[j][i][k] += int((1 - delX) * delY * src_image[yBase + 1][xBase][k])
 
                    if (xSrc < (xDim - 1)) and (ySrc < (yDim - 1)):
                        dest_image[j][i][k] += int(delX * delY * src_image[yBase + 1][xBase + 1][k])
 
    return dest_image
 
 

if __name__=="__main__":
    main()



