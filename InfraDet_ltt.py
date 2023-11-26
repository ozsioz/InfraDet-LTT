# -*- IR Thermography Laminar-Turbulent Transition and Flow Separation Detection tool*-
"""
Created (2022) , modified to the latest version on 24 November 2023

@author: Özge Sinem Özçakmak
ozgesinemozcakmak@gmail.com
Reference:
https://github.com/ozsioz/InfraDet-LTT
MIT License
"""
###########################################################################################################################################
## !!!! The code is executed from the Anaconda Prompt
#    1. First open the directory where this code is saved in anaconda promt, (Example: cd Users/.../Desktop/IRFile/Re5mil_corrected)
#    2. Then execute this command: 'python InfraDet_ltt.py calibrationIR_upper.dat'
#    3. The code will ask the user to choose where the IR camera experiment images are located. 
#    4. Once the folder is seleted, the transition and sepration codes work and create images in the same folder with the same file name ending with -tr and -sp.

# This python script reads corrected infrared thermography images, crops and enhances,
# finds transition and separation locations, marks on the image and saves the data
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog
from scipy.signal import find_peaks_cwt
import sys

left_sep=0 # offset margin to start scanning the separation region after the transition location

def main():

    global filename, path, fileout, counter, fint, inputfile
    global xOcList, curveFit_mList, curveFit_nList
    global left, right, top, bottom
    global thresh01s, threshpeaksep , thresh01p, threshSigmaTr, threshSigmaSp, threshPixelTr, threshPixelSp, threshStddevTr, threshStddevSp, maxratiosep,maxvalue_sep
 # Select the folder with thermography images
    Mbox('Image Location', 'Please select the folder where images are located', 0)
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askdirectory()

###############################################################################################
##  THESE ARE THE PAREMETERS TO CHANGE ACCORDING TO THE IMAGE , IR camera quality, pixel intensity, crop boundaries etc. #####

# Parameters to check the peaks of locally found transition and separation points.

    thresh01s = 0.4 # peak ratio (between 0 and 1 )  # Higher value means more clean data but may cause loss of information such as the locations of transition and separation locations.
    maxvalue_sep=60  # limit for the magnitude of seperation peaks  # 40

# Parameters for validity check around average (transition and separation),
 
    threshSigmaTr = 0.8 # threshold for distance between detected points
    threshSigmaSp = 0.6 # Increase in this number will create more detected numbers but more scattered data 
    # When it is too scattered it can fail due to second criteria of the threshStddevTr (standard deviation of the detected points) Try to adjust both together to find the best results
    
# Parameters to check the std deviation levels of spanwise transition/separation distributions
    threshStddevTr = 24   # 24
    threshStddevSp = 24   # 20 is the default number, increasing this threshold would end up in more detected points.

# Parameters to check number of valid spanwise points
    threshPixelTr = 6   # 6, 8, 12, 15 can be tried.
    threshPixelSp = 15  # default 15

#############################################################################################


# Open the output data file
    fileout = open('tr-out.dat', 'w')
    fileout.write(
        'Run No \t \t \t  \t \t AOA \t xtr_pixel \t \t x_tr/c \t xsp_pixel \t \t x_sp/c \n')

# Read the commandline input (name of the input file that contains calibration data and crop boundary information)
    inputfile = sys.argv[1]
    filein = open(inputfile, 'r')
# Read the file line by line and fill all data into the list "curveFit"
    curveFit = filein.readlines()
    ccf1 = np.array(curveFit)  # convert the list to an numpy array
    ccf2 = ccf1.astype(np.float64)  # convert to float
    print(ccf2, "ccf2")
# Number of chordwise position data in the calibration images
    noChordPos = ccf2[0]

    xOcList = []  # List of chordwise calibration point positions
    curveFit_mList = []  # List of m coefficients for the linear curvefits for each chordwise position that calibration is performed
    curveFit_nList = []  # List of n coefficients for the linear curvefits for each chordwise position that calibration is performed

# Fill up the above lists from the data read from the input file
    for inoChordPos in range(0, int(noChordPos)):
        xOc = ccf2[3*inoChordPos+1]
        curveFit_m = ccf2[3*inoChordPos+2]
        curveFit_n = ccf2[3*inoChordPos+3]
        print(curveFit_m, "Curvefit_m")
        print(curveFit_m, "Curvefit_n")
        xOcList.append(xOc)
        curveFit_mList.append(curveFit_m)
        curveFit_nList.append(curveFit_n)

# Create the crop boundaries as read from the input file
    left = int(ccf2[-4])
    right = int(ccf2[-3])
    top = int(ccf2[-2])
    bottom = int(ccf2[-1])

# For consistency, it is adviced having the pressure side of the airfoil mirrored adn using always the 'upper' option for both. So, they have the same direction as the suction side (leading edge on the left side of the image
# in order to eliminate the order confusion in the crop boundaries) Therefore the below lines can kept as a comment
    # if 'lower' in inputfile:
    #     temp = left
    #     left = right
    #     right = temp

# Scan all images in the selected folder and analyze them one by one.
    counter = 0  # counter for analyzed number of images
    for filename in os.listdir(path):
        print('-----------------')
        if filename.endswith('.tiff'):  # only works on tiff images
            counter += 1
            print(filename)
            imageSelect = os.path.join(path, filename)

# TRANSITION DETECTION by the find_transition function    
            (max_levsp, (fname, AoA, mean_Trx2, xTrChord)) = find_transition(imageSelect) 
          
# Choose file that is edited by find_transition function
            filename_edited_by_trans = filename.replace(".tiff", "-tr.tiff")
            
# Choose the image based on the file edited by find_transition function
            imageSelectSeparation = os.path.join(path, filename_edited_by_trans)
            
# SEPARATION DETECTION by the find_separation function 
          
            (mean_Spx2, xSpChord) = find_separation(imageSelectSeparation, max_levsp+20)    #Here the user can manually add more distace on the input for finding separation
#   Above maxlevsp+3 --> 3 is added to give some distance after the detected transition line so that the programme does not catch the adjacent 
#   pixel intensity peaks related to the transition line, and instead finds the clean separation line. This value can be changed according to the images)
            
            xTrChord_dec=float("{0:.3f}".format(xTrChord)) 
            xSpChord_dec=float("{0:.3f}".format(xSpChord))

            mean_Trx2_dec=float("{0:.3f}".format(mean_Trx2))
            mean_Spx2_dec=float("{0:.3f}".format(mean_Spx2))
            
# Write the detected spanwise average pixel and x/c locations for each image to a text data file.
            fileout.write('%s \t  %s \t %s \t \t \t %s \t \t %s \t \t %s \n' % (fname, AoA,str( mean_Trx2_dec).rstrip(), str(xTrChord_dec).rstrip(), str(mean_Spx2_dec).rstrip(),str(xSpChord_dec).rstrip()))
          
#########################################################################################

#%%
def find_transition(imageSelect):

    global min_value, max_value, min_value2, max_value2,filename

# Read input image
    img = cv2.imread(imageSelect)
# Convert to grayscale
    imgGray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
## !! The left crop boundary of the image
    leftn=left+5     # Here you can change the crop boundary according to the angle of attack as at some high or very low angles like 20 or -20 degrees,
                     # you might loose some chorswise locations using the predetermined crop boundaries from the calibration 
                     # The common choices are 5,-5 or 0.
                     
# Crop image using the crop boundaries read from the input file (calibrationIR_upper.dat )
    imgGray2 = imgGray1[top:bottom, leftn:right]

    # Enhance cropped image using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(2, 2))
    imgGray = clahe.apply(imgGray2)
    
   # Show the cropped image for check
    # cv2.imshow('test', imgGray)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

# Get row and column information from the image
# Transition finder works on this cropped image
    rows, cols = imgGray.shape

    import operator  # needed later below for local maxima finding.

    # Range for scanning pixel columns
    cStart = 0
    cEnd = cols

    # Range for scanning pixel rows
    rowBegin = 0
    rowEnd = rows

    # Number of local pixel rows to be analyzed and assigned one transition location
    step = 3

    # Keep the original image as newImage_color. Later it will marked with transition positions.
    newImage_color = img

    # Parameters for spanwise average pixel location for transition and separation
    Trx_local = []

    Try_local = []

# Main loop that scans the pixel rows.
    # Scans every "step" pixel rows and assigns one value of
    # transition/separation location to that "step" number of pixel rows.
  
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
  

    
    for rr in range(rowBegin, rowEnd-step, step):
# Parameters to hold the found transition/separation locations on the -to be scanned- "step" number of pixel rows
        ffTr = []

# Start and end ranges for the "step" number of pixel rows
        rStart = rr
        rEnd = rr+step-1

# Start scanning the "step" number of pixel rows
        for jj in range(rStart, rEnd):

# f is the variation of pixel intensity along the columns of the currently scanned pixel row
            scanList_ss = imgGray[jj][cStart:cEnd]
            scanList_ps = scanList_ss[::-1]

# Scan from left to right if it is upper (suction) side, else scan from right to left, 
# It is strongly adviced using the mirrored image for pressure side and choosing it also as upper side. Otherwise the commented if 'lower' should be activated.
            if 'upper' in inputfile:
                f = np.array(scanList_ss, dtype=np.float64)
            # if 'lower' in inputfile:
            #     f = np.array(scanList_ps, dtype=np.float64)

            # MOVING AVERAGE PLOT
            # plt.plot(moving_average(f, n=10))
            # plt.show()

            # p is the moving average of the gradient of the moving average of f.
            # (i.e. first f is moving averaged. Then its gradient is found and then moving averaged)
            # This moving averaging is needed because the variations of f and p -if not averaged- are very patchy
            # 5 is the moving averaging range. Smaller values make it closer to actual visualized transition line.
            # However since it is less averaged, finding peaks becomes harder.
            
            p = moving_average(np.gradient(moving_average(f, n=2)), n=6) # Defaults n=2 n=5

            # Absolute value of p is needed to detect separation
            # Because the gradient is high negative if there is separation whereas it is high positive for transition.
            p = abs(p)

            p = p**5  
            # plt.plot(p)
            # plt.show()

            # Find the pixel locations (indices) of the peaks in p within a local pixel neighborhood of 50,100 or 200 this value can be changed according to the case.
            indexes = find_peaks_cwt(p, np.arange(1, 200))

            # Find the actual value of p where it has peaks.
            pp = [p[i] for i in indexes]

            # Find the first maximum among the list of peaks, which can be the transition location depending on pixel position
            max1_index, max1_value = max(
                enumerate(pp), key=operator.itemgetter(1))

            # Set the max value to -999 in order to find the second maximum below.
            pp[max1_index] = -999

            # Find the second maximum among the list of peaks, which can be the separation location depending on pixel position
            max2_index, max2_value = max(
                enumerate(pp), key=operator.itemgetter(1))

            # Put the maximum 1 value back into its original position. 
            pp[max1_index] = max1_value
            # print('filename max1 max 2',(filename,max1_value,max2_value))
            # print (indexes[max1_index],indexes[max2_index])
            # Assign the local transition locations on the currently scanned pixel row

            if 'upper' in inputfile:
               # meaning there are two relatively strong peaks pointing to the existence of both transition and separation.
               if max1_value>20 : # default =20 , no need to change this number usually.
                   # meaning if the first maximum occurs earlier on the pixel row, when scanned from left to right (i.e. SS).
                    if indexes[max1_index] < indexes[max2_index]:
                           xTr = indexes[max1_index] + leftn
                    else:
                           xTr = indexes[max2_index] + leftn

               else:
                   xTr = indexes[max2_index] + leftn

   # ffTr is a list that contains local transition locations along each pixel row within the currently scanned 'step' number of rows
            ffTr.append(xTr)
          
   # Generate the histogram of transition locations using the step number of pixel rows
   # and find the pixel location that has the maximum of the histogram
   # then assign this pixel location as the local transition location found for this row block that consists of step number of pixel rows
        nTr, bTr = np.histogram(ffTr, bins=100)
        elemTr = np.argmax(nTr)
        bin_maxTr = bTr[elemTr]
        localTr = int(bin_maxTr)  # local transition location

   # These two lists contain local transition locations in pixels at each
   # x (chordwise, or pixel columns) and y (spanwise, or pixel rows) location
        Trx_local.append(localTr)
        Try_local.append(rStart+top)

  
  # End of the loop for rows

 ########################## TRANSITION #####################################

  # Calculate the average and std dev of estimated spanwise transition locations
    Trx = np.array(Trx_local, dtype=np.float64)
    averageTrx = np.mean(Trx)
    stddevTrx = np.std(Trx)

  # Set the sigma value for validity check around average (transition)
    sigmaTr = threshSigmaTr*stddevTrx
    
  # Calculate the histogram of spanwise distribution of the local transition locations
  # then find the maximum of the histogram
  # then assign the maximum value as the temporary spanwise average location of the transition location
    nTr1, bTr1 = np.histogram(Trx_local, bins=100)
    elemTr1 = np.argmax(nTr1)
    bin_maxTr1 = bTr1[elemTr1]
  # spanwise average transition location (temporary)
    Tr1_max_span= int(bin_maxTr1)
    print("Tr1_max_span", Tr1_max_span, "\n\n")

    """"FILTERING"""
  # Now check how far each of the local transition locations away from the spanwise average
    Trx_local2 = []
    Try_local2 = []
    for k1 in range(0, len(Trx_local)):
        # if the local transition location is more than sigma away then disregard that point. This provides some filtering of the data
        if abs(Trx_local[k1]-Tr1_max_span) < sigmaTr:
            Trx_local2.append(Trx_local[k1])
            Try_local2.append(Try_local[k1])

  # Calculate the histogram of filtered spanwise distribution of the local transition locations
  # then find the maximum of the histogram
  # then assign the maximum value as the actual spanwise average location of the transition location
    nTr2, bTr2 = np.histogram(Trx_local2, bins=100)
    elemTr2 = np.argmax(nTr2)
    bin_maxTr2 = bTr2[elemTr2]
    Tr1_max_span_filtered = int(bin_maxTr2)  # spanwise average transition location (actual)
    Trx2 = np.array(Trx_local2, dtype=np.float64)
    stddevTrx2 = np.std(Trx2)

    try:
        mean_Trx2 =int(np.mean(Trx2))
    except:
        mean_Trx2=999
    leftseplist=[]
   
   ############################# THRESHOLDS ##################################################

    if stddevTrx2 > threshStddevTr:
        mean_Trx2 = 0
        print('caayse stddevTrx2 > threshStddevTr')

   # If number of filtered data points are less than a certain value probably there is no valid transition location
    if len(Trx_local2) < threshPixelTr:  # eleman sayisi
        mean_Trx2 = 0
        print('cause len(Trx_local2) < threshPixelTr eleman sayisi')
  
###########################################################################
  # Mark the local transition locations on the original but corrected image
    if Tr1_max_span_filtered != 0:
        for k3 in range(0, len(Trx_local2)):
            leftseplist.append(int(Trx_local2[k3])) #
            
            cv2.line(newImage_color,
                     (int(Trx_local2[k3]), int(Try_local2[k3])),
                     (int(Trx_local2[k3]), int(Try_local2[k3]+step)),
                     (0, 0, 255), 2)
            
    left_sep=max(leftseplist)    

  # Find the AoA value by scanning the names of the files that are currently being analyzed.
    AoA2 = float(AoA)
    print(AoA2, 'AoA2')
    pixelList = []

  # Create the pixel list corresponding to the chordwise positions from the read calibration data
    for iChordList in range(0, len(xOcList)):
        P = curveFit_mList[iChordList]*AoA2 + curveFit_nList[iChordList]
        pixelList.append(P)

  # Find the curvefit coefficients for the current angle of attack...pixel vs x/c
    curveFit6_m, curveFit6_n = np.polyfit(pixelList, xOcList, 1)

  # Using the calculated curvefit coefficients calculate the x/c location of the transition location
    if mean_Trx2 == 0:
        xTrChord = 0
    else:
        xTrChord = curveFit6_m*mean_Trx2+curveFit6_n

  # Write the found pixel and x/c locations for each image on the images as well
    def print_angle_of_attack_label():
        cv2.putText(newImage_color, "AoA= %s" % (str(round(AoA2, 2))),
                    (380, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (61, 241, 228), 1)

    print_angle_of_attack_label()

    def print_transition_label():
        if xTrChord == 0:
            cv2.putText(newImage_color, "Transition not detected",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            cv2.putText(newImage_color, "Transition: %s%%" % (
                str(round(xTrChord, 2))), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    print_transition_label()
    

  # Save the marked images in "/tr" folder, also add "-tr" to the marked image name
    filename2 = filename[:-5]
    filename2 = filename2 + "-tr.tiff"
    path2 = path.replace('processed', 'tr')
    if not os.path.exists(path2):  # if the "tr" folder does not exist create it
        os.makedirs(path2)

    filename3 = os.path.join(path2, filename2)
    cv2.imwrite(filename3, newImage_color)  
    
    return (left_sep, (filename[-24:-5], AoA, mean_Trx2, xTrChord))

#%% SEPARATION DETECTION
############################################################################################################
def find_separation(imageSelectSeparation,left_sep):
    
    global min_value, max_value, min_value2, max_value2

  # Read input image
    img = cv2.imread(imageSelectSeparation)

  # Convert to grayscale
    imgGray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print('left_sep', left_sep)
  
  # Crop image using the crop boundaries read from the input file. !!! Here 20 pixels is added to give a distance from the right side. This depends on the image anc can be adjusted.
    imgGray2 = imgGray1[top:bottom, left_sep:right+20]

  # Enhance cropped image using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(2, 2))
    imgGray = clahe.apply(imgGray2)

  # Get row and column information from the image
  # Separation finder works on this cropped image
    rows, cols = imgGray.shape

    import operator  # needed later below for local maxima finding.

  # Range for scanning pixel columns
    cStart = 0
    cEnd = cols

  # Range for scanning pixel rows
    rowBegin = 0
    rowEnd = rows

  # Number of local pixel rows to be analyzed and assigned one transition location
    step = 3

  # Keep the original image as newImage_color. Later it will marked with transition positions.
    newImage_color = img

  # Parameters for spanwise average pixel location for transition and separation
    Sp_loc_pix_x = []

    Sp_loc_pix_y = []
  
  # Main loop that scans the pixel rows.
  # Scans every "step" pixel rows and assigns one value of
  # transition/separation location to that "step" number of pixel rows.
    for rr in range(rowBegin, rowEnd-step, step):
  # Parameters to hold the found transition/separation locations on the -to be scanned- "step" number of pixel rows
        ffSp = []
        max2all=[]
        max1all=[]

  # Start and end ranges for the "step" number of pixel rows
        rStart = rr
        rEnd = rr+step-1

  # Start scanning the "step" number of pixel rows
        for jj in range(rStart, rEnd):

  # f is the variation of pixel intensity along the columns of the currently scanned pixel row
            scanList_ss = imgGray[jj][cStart:cEnd]
            scanList_ps = scanList_ss[::-1]

  # Scan from left to right if it is upper (suction) side, else scan from right to left. It is suggested that pressure side is mirrored and only upper side is used.
            if 'upper' in inputfile:
                f = np.array(scanList_ps, dtype=np.float64)
            # if 'lower' in inputfile:
            #     f = np.array(scanList_ps, dtype=np.float64)

  # Absolute value of p is needed to detect separation
  # Because the gradient is high negative if there is separation whereas it is high positive for transition.
            p = f
            p = p**5

  # Find the pixel locations (indices) of the peaks in p within a local pixel neighborhood of 40
            indexes = find_peaks_cwt(p, np.arange(1, 40))

  # Find the actual value of p where it has peaks.
            pp = [p[i] for i in indexes]

  # Find the first maximum among the list of peaks, which can be the transition location depending on pixel position
            max1_index, max1_value = max(
                enumerate(pp), key=operator.itemgetter(1))

  # Set the max value to -999 in order to find the second maximum below.
            pp[max1_index] = -999

  # Find the second maximum among the list of peaks, which can be the separation location depending on pixel position
            max2_index, max2_value = max(
                enumerate(pp), key=operator.itemgetter(1))

  # Put the maximum 1 value back into its original position. 
            pp[max1_index] = max1_value
           
            if max1_value>maxvalue_sep:
                if indexes[max1_index]<indexes[max2_index]:  #if max1 is earlier from right side
                    xSp = -indexes[max2_index] + right 
                else:
                    xSp = -indexes[max1_index] + right
            else:
                xSp = 0  
  # ffTr is a list that contains local transition locations along each pixel row within the currently scanned 'step' number of rows
            ffSp.append(xSp) 
       

  # Generate the histogram of separation locations using the step number of pixel rows
  # and find the pixel location that has the maximum of the histogram
  # then assign this pixel location as the local separation location found for this row block that consists of step number of pixel rows
        max2all.append(max2_value)
        max1all.append(max1_value)

        nSp, bSp = np.histogram(ffSp, bins=100)
        elemSp = np.argmax(nSp)
        bin_maxSp = bSp[elemSp]
        localSp = int(bin_maxSp)  # local separation location
        
  # These two lists contain local separation locations in pixels at each
  # x (chordwise, or pixel columns) and y (spanwise, or pixel rows) location
  # Note that if there is no separation the location is not added to the list.
      
        if localSp != 0:
            Sp_loc_pix_x.append(localSp)
            Sp_loc_pix_y.append(rStart+top)
       
  # End of the loop for rows

############################## SEPARATION  #################################################
   
 # Calculate the average and std dev of estimated spanwise separation locations
    Spx = np.array(Sp_loc_pix_x, dtype=np.float64)
    averageSpx = np.mean(Spx)
    stddevSpx = np.std(Spx)

 # Set the sigma value for validity check around average (separation)
    sigmaSp = threshSigmaSp*stddevSpx

 # Calculate the histogram of spanwise distribution of the local separation locations
 # then find the maximum of the histogram
 # then assign the maximum value as the temporary spanwise average location of the separation location
    nSp1, bSp1 = np.histogram(Sp_loc_pix_x, bins=100)  # Default: 100
    elemSp1 = np.argmax(nSp1)
    bin_maxSp1 = bSp1[elemSp1]
 # Spanwise average separation location (temporary)
    aveSp1 = int(bin_maxSp1)
    print('nSp1',nSp1)
    print('aveSp1 location',aveSp1)
    print('elemSp1',elemSp1)
    print('maxnsp1',max(nSp1))
    
 # Checking the distance between the local separation locations and the spanwise average
    Sp_loc_pix_x2 = []
    Sp_loc_pix_y2 = []
    for k2 in range(0, len(Sp_loc_pix_x)):
 # if the local separation location is more than sigma away then disregard that point. This provides some filtering of the data
        if abs(Sp_loc_pix_x[k2]-averageSpx) < sigmaSp:
            Sp_loc_pix_x2.append(Sp_loc_pix_x[k2])
            Sp_loc_pix_y2.append(Sp_loc_pix_y[k2])

  # Calculate the histogram of filtered spanwise distribution of the local separation locations
  # find the maximum of the histogram
  # then assign the maximum value as the actual spanwise average location of the separation location
    nSp2, bSp2 = np.histogram(Sp_loc_pix_x2, bins=100)
    elemSp2 = np.argmax(nSp2)
    bin_maxSp2 = bSp2[elemSp2]
    aveSp = int(bin_maxSp2)  # spanwise average transition location (actual)
    
    Spx2 = np.array(Sp_loc_pix_x2, dtype=np.float64)
    stddevSpx2 = np.std(Spx2)
    try:
        mean_Spx2 =int(np.mean(Spx2))
    except:
        mean_Spx2=999
  
############################# THRESHOLDS ################################################# 
    if stddevSpx2 > threshStddevSp:  # <----- 3. cause
        print("cause 3: stddevSpx2 > threshStddevSp")
        print("stddevSpx2: ", stddevSpx2)
        print("threshStddevSp: ", threshStddevSp, "\n\n")
        mean_Spx2 = 0
    
    
    deltaCheck = abs(left_sep-mean_Spx2)
    if deltaCheck < 30:  # <----- 1. cause
        print("cause 1: deltaCheck < 30")
        mean_Spx2  = 0
 # If number of filtered data points are less than a certain value probably there is no valid separation location

    if len(Sp_loc_pix_x2) < threshPixelSp:  # <----- 2. cause
        print("cause 2: len(Sp_loc_pix_x2) < threshPixelSp")
        print("len(Sp_loc_pix_x2): ", len(Sp_loc_pix_x2))
        print("threshPixelSp: ", threshPixelSp, "\n\n")
        mean_Spx2 = 0
 
 # Mark the local separation locations on the original but undistorted image
    if mean_Spx2 != 0:
        for k4 in range(0, len(Sp_loc_pix_x2)):
            cv2.line(newImage_color, (int(Sp_loc_pix_x2[k4]), int(Sp_loc_pix_y2[k4])), (int(
                Sp_loc_pix_x2[k4]), int(Sp_loc_pix_y2[k4]+step)), (0, 255, 0), 2)
      
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
   
    AoA2 = float(AoA)
    print(AoA2, 'AoA2')
    pixelList = []

  # Create the pixel list corresponding to the chordwise positions from the read calibration data
    for iChordList in range(0, len(xOcList)):
        P = curveFit_mList[iChordList]*AoA2 + curveFit_nList[iChordList]
        pixelList.append(P)

  # Find the curvefit coefficients for the current angle of attack...pixel vs x/c
    curveFit6_m, curveFit6_n = np.polyfit(pixelList, xOcList, 1)

  # Using the calculated curvefit coefficients calculate the x/c location of the separation location
  # only if there "is" separation
   
    if mean_Spx2 == 0:
        xSpChord = 0
    else:
        xSpChord = curveFit6_m*mean_Spx2+curveFit6_n
   
    # Write the found pixel and x/c locations for each image on the images as well
    def print_angle_of_attack_label():
        cv2.putText(newImage_color, "AoA= %s" % (str(round(AoA2, 2))),
                    (380, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (61, 241, 228), 1)

    print_angle_of_attack_label()

    def print_separation_label():
        if xSpChord == 0:
            cv2.putText(newImage_color, "Separation not detected",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(newImage_color, "Separation: %s%%" % (
                str(round(xSpChord, 2))), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    print_separation_label()

    # Save the marked images in "/tr" folder, also add "-tr" to the marked image name
    filename2 =  filename[:-5]
    filename2 = filename2 + "-sp.tiff"
    path2 = path.replace('processed', 'tr')
    if not os.path.exists(path2):  # create the "/tr" folder if does not exist
        os.makedirs(path2)

    filename3 = os.path.join(path2, filename2)
    cv2.imwrite(filename3, newImage_color)
    
    return (mean_Spx2, xSpChord)

 
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def Mbox(title, text, style):
    print(title, "\n", text, "\n", style, "\n\n")


if __name__ == "__main__":
    main()
