# -*- IR Thermography Image Calibration pixels versus chordwise positions on the model*-
"""
Created the latest version on 24 November 2023

@author: Özge Sinem Özçakmak
ozgesinemozcakmak@gmail.com
Reference:
https://github.com/ozsioz/InfraDet-LTT
MIT License
"""
###########################################################################################################################################
## !!!! The code is executed from the Anaconda Prompt
#    1.First open the directory where this code is saved in anaconda promt, (Example: cd Users/.../Desktop/IRFile/calibrationfile)
#    2. Then execute this command: 'python InfraDet_calibration.py'
#    3. The code will ask the user to choose the files where the corrected calibration images located. Choose the location.
#    4. The code asks is it upper or lower side of the airfoil (example: ' upper')
#    5. The code asks the calibration chord locations from 0 to 100 (chordwise percantages). (Example: '0,10,20,30,40,50,60,70,80,90,100')
#    6. If you named the images as in he example images the code will automatically read the angle of attack information from the file names.
#    7. The user needs to click the chordwise locations on the image that corrresponds to 0,10,20,30.. as taped on the model during image acuisition)
#    Here, the user should click exactly the same amount clicks corresponding to the locations entered in Step 5!
#    The code generates an output file calibrationIR_upper.dat which then should be copied to the file containing the infradet_ltt.py
#########################################################################################################################################
## This code creates an output file calibrationIR_upper.dat, which is used in transition/separation detection tool infradet_ltt.py

import cv2
import numpy as np
import os 
from win32api import GetSystemMetrics
import ctypes
import tkinter as tk
from tkinter import filedialog 
import sys


def main ():
    global filename,path,fileout,counter,fint,AoA,left_clicks,clickCounter,list3
    
  ## Select the folder with the IR thermography images
    Mbox('Image Location', 'Please select the folder where images are located', 0)
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askdirectory()
    left_clicks = list()  
    counter = 0 # counter for analyzed number of images
    
    ##Read commandline inputs: 1) chordwise positions at which the model is marked 2)upper or lower
    airfoilside=input('Upper or lower side of the airfoil? write upper or lower   ')
    list1 = input('Enter enter calibration chord locations from 0 to 100 in separated format ') # 0 10 20 30 40 , having a zero chord location is extremely suggested.
    print(type(list1))
    listt=list1.split(',')
    list2 = np.array(listt)
    list3 = list2.astype(np.float64)
    print(list3,"list3 ")
    uplow = airfoilside
       
    P = [] #List of pixel positions for all images (i.e. for all angles of attack). Each angle of attack is a separate sublist.
    C = [] #List of chordwise positions for all images (i.e. for all angles of attack). Each angle of attack is a separate sublist.
    
    AoA3    = [] #List of angles of attack

  ##Start reading each recorded calibration image one by one in the selected folder.
    for filename in os.listdir(path):
        if filename.endswith('.tiff'): ##scan for tiff files only.
            counter += 1
            left_clicks[:] = [] #counter for left clicks on the image
            
   ##Find the AoA value from the file name
            
            if  filename[-9] == '_':
                
                if filename[-8] == '0':
                    AoA = '%s' %(0.0)
                else:    
                    AoA = '%s' %(filename[-8])
                
            elif filename[-10] == '_':
                
                if filename[-9] == '-':
                    AoA = '-%s' %(filename[-8])
              
                else:
                    AoA = '%s' %(filename[-9]+filename[-8])
                    
            elif filename[-10] == '-':
                AoA = '-%s' %(filename[-9]+filename[-8])  
            
            else:
                print ('not correct format')
                
            print (AoA,"AOA ")
            
            AoA3.append(AoA) ##fill up the AoA list
            imageSelect = os.path.join(path,filename)
            getCal(imageSelect) ##open up current image and record the left and right clicks on the image, i.e. calibration markings on the image.
            
            array = np.array(left_clicks) #array of all left clicks
            chordPos = array[:,0] #array of clicked chordwise positions
            chordPos = chordPos.astype(np.float64) #convert to float
            print (chordPos,"chordPos ") # 0,10,20...
            pixelPos = array[:,1] #array of clicked pixel positions
            pixelPos = pixelPos.astype(np.float64) #convert to float
            print (pixelPos,'pixelpos') #the pixels in the x axis
            AoA2     = array[:,3] #array of angles of attack
            AoA2     = AoA2.astype(np.float64)
            print(array,"Array ") # 0,x,y,aoa    20,x,yaoa
            ##Get the number of row and column information from the image
            img2 = cv2.imread(imageSelect)
            img2Gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)   #color conversion (inputimage,flag), flag=gray convesion
            rows2,cols2 = img2Gray.shape #get the image size
            print(rows2,"rows2")
            print(cols2,"cols2")

            #Fill up P and C as defined before
            P.append(pixelPos) # adds element to the end of list (append) only for x positions
            C.append(chordPos)    #inputs 0,20...
            print(P,'P')
    array_AoA3 = np.array(AoA3)
    array_AoA3 = array_AoA3.astype(np.float64)
    
    #Open and start writing to the calibration data output file which will be an input for infradet_ltt.py ,  transition detection code.
    outfilename = 'calibrationIR_'+uplow+'.dat'
    outfile = open(outfilename, 'w')
    outfile.write('%f \n' %(len(list3))) #write the number of chordwise positions
    
   ##Construct the pixel list and the AoA list in order to create AoA vs pixel data for each chordwise position
    for iChord1 in range(0,len(list3)):
        pixelList = []
        aoaList = []
        for  iAoA in range(0,counter):
            if len(C[iAoA][:])==len(list3):
                pixelList.append(P[iAoA][iChord1])
                aoaList.append(array_AoA3[iAoA])
            else:
                for iChord2 in range(0,len(C[iAoA][:])):        
                    if C[iAoA][iChord2] == list3[iChord1]:
                        pixelList.append(P[iAoA][iChord2])   # for all the angles
                        aoaList.append(array_AoA3[iAoA])
   ##Then fit linear curves for each chordwise position (AoA vs pixel)
   ##and write output to the file, i.e. linear curvefit coefficients for each chordwise position
        print(pixelList,'pixellist')
        print(np.isfinite(aoaList).all(),'is it')
        print(np.isfinite(pixelList).all(),'is it pix')
        curveFit_m, curveFit_n = np.polyfit(aoaList,pixelList,1)
        outfile.write('%f \n' %(list3[iChord1]))
        outfile.write('%f \n' %(curveFit_m))
        outfile.write('%f \n' %(curveFit_n))
                
    ##Find the crop boundaries (! The user needs to adjust here according to the first and the last position entered in the prompt window)
    ichord0,  = np.where(chordPos==list3[0])   # returns an array that fits to a condition 
    ichord50, = np.where(chordPos==list3[-1])
    print(list3[-1],"list3 [-1]")
    
  ######### !!!! The Offsets here should be changed according to the image dimensions and the interested area  ######
  ######## This area should cover all the images at different angle of attack values. You should be able to see the leading edge and the intrested model area)  !!!!!!!!!!!!!!!!!!!  ####### 
    offset =   20 #left
    offset2 = 10 #right 
    offset3 = 80  #bottom
    offset4= 100 #top
    offset5=10
    
    ## Applying offsets and calculating pixel-chord locations
    
    if len(ichord0)<(counter):  # If the leading edge is not clicked at every image
    
        ichordsecond, = np.where(chordPos==list3[1])     # choose second position
        pixelsecond = []
        print(pixelPos,"pixelpos")
        for index in ichordsecond:
            print(index,"index")
            pixelsecond.append(pixelPos[index])
        print(pixelsecond,"pixelsecond")    
        pixelsecond_a = np.array(pixelsecond)
        print(pixelsecond_a,"pixelsecond_a")
        left   = int(pixelsecond_a.max() - offset    )   
    
    elif len(ichord0)==(counter):   # If in every image leading edge is seleected in calibration
        ichordsecond, = np.where(chordPos==list3[1])     #second pozisiton 
        pixelsecond = []
        pixel0=[]
        for index in ichord0:
            pixel0.append(pixelPos[index])
        pixel0_a = np.array(pixel0)
        print(pixelPos,"pixelpos")
        for index in ichordsecond:
            print(index,"index")
            pixelsecond.append(pixelPos[index])
        print(pixelsecond,"pixelsecond")    
        pixelsecond_a = np.array(pixelsecond)
        print(pixelsecond_a,"pixelsecond_a")
        left   = int(pixel0_a.max())

    top = offset4
    
    pixellast = []
    for index in ichord50:
        pixellast.append(pixelPos[index])
    pixellast_a = np.array(pixellast)
    
    right = int(pixellast_a.max() + offset2)
    
    bottom = rows2-offset3
    print(top,'top')
    print(bottom,'bottom')
    print(left,'left')
    print(right,'right')
    
# Write the crop boundaries to the output file
    outfile.write('%f \n' %(left))
    outfile.write('%f \n' %(right))
    outfile.write('%f \n' %(top))
    outfile.write('%f' %(bottom))
 
    imgGray2 = img2Gray[top:bottom, left:right]

# Enhance cropped image using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(2, 2))
    imgGray = clahe.apply(imgGray2)
    cv2.imshow('test', imgGray)
    cv2.waitKey()
    cv2.destroyAllWindows()

# Function for calibration    
def getCal(imageSelect):
  
    #global left_clicks
    global clickCounter
    
    clickCounter = 0
    
 ## Read input image
    img = cv2.imread(imageSelect)
    
    width = GetSystemMetrics(0)
    height = GetSystemMetrics(1)
    scale_width = 480 / img.shape[1]
    scale_height = 640 / img.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)
    cv2.namedWindow(filename, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(filename, window_width, window_height)

#set mouse callback function for window
    cv2.setMouseCallback(filename, mouse_callback)
    dx, dy = 20,20

# Custom (rgb) grid color
    grid_color = [154,205,50]

# Modify the image to include the grid
    img[:,::dy,:] = grid_color
    img[::dx,:,:] = grid_color
    
    cv2.imshow(filename, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def mouse_callback(event, x, y, flags, params):
    global clickCounter
    
#If the mark on the image is visible then left click, if its not visible right click.
#Number of total left and right clicks should be equal to the commandline input number of chordwise positions
    if event == cv2.EVENT_LBUTTONDOWN:      
        
        clickCounter += 1
        
#store the coordinates of the left-click event
        left_clicks.append([list3[clickCounter-1], x, y, AoA]) ##also store the current clickcounter and AoA information

    elif event == cv2.EVENT_RBUTTONDOWN: ##right click event means there is no calibration data
        clickCounter += 1

def Mbox(title, text, style):
    ctypes.windll.user32.MessageBoxW(0, text, title, style)
    
if __name__=="__main__":
    main()




