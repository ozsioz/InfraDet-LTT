This tool was built for detecting the physical phenomena of laminar to turbulent transition (LTT) of the flow on a wind turbine blade or on aircraft wing sections from Infrared Thermography camera images. The tool consist of 4 codes for 4 steps:
1. Image manipulation for camera angle correctiona dn lens distortion correction [IRCam_Imagecorrection.py]
2. Image calibration for different angle of attack values (the translation between chordwise physical positions to pixels on the image for each angle of attack)  [InfraDet_calibration.py]
3. LTT detection for positive angle of attack values [InfraDet_ltt.py]
4. LTT detection for negative angle of attack values [InfraDet_ltt_neg.py]


## IRCam_Imagecorrection.py
All the calibration and experiment imahges should be corrected with the same code settings

## InfraDet_calibration.py
This code transfers image pizel positions to the physical chorswise positions on the model for each angle of attack value, creates coefficients for the and it creates an output file calibrationIR_upper.dat, which is used in transition/separation detection tool infradet_ltt.py

## calibrationIR_upper.dat
This output file is obtained for chorswise locations of 0,10,20,30,40,50,60,70,80,90 % on the model for various angle of attacks (e.g. using more files than uploaded in this demo, so it is more accurate than the one the user would obtained with three images)
