# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 12:16:38 2024

@author: ThomasZimmerman
"""
DEBUG=0     #1=print debugging text to monitor
DEBUG_Z=1   #1=print Z and Gabor values
TRACK_RECO_ENABLE=1     #0=disable track and reco to speed up fram processing when adjusting detection values with keyboard interface
ENABLE_RECO=1
DISPLAY_REZ=(640,480)
#MAX_PARM=7
#(XC,YC,ZC,ID,STILL_COUNT,AREA,STATE)=range(MAX_PARM)

# Reconstruction math
dxy   = 1.4e-6 # imager pixel size in meters.
wvlen = 650.0e-9 # Red 
#wvlen = 405.0e-9 # Blue 
zScale=1e-6 # convert z units to microns 

# Tracking
W=80 # increase crop window to capture fringes of raw hologram
IMAGE_SCALE=2
MAX_OBJ=10
MAX_DETECT=10
MAX_TRACK=10
MAX_STILL=5
MAX_DISTANCE=500 # if detected and tracked obj exceed this distance, don't let tracker match them
MAX_PARM=13
(FRAME,ZC,X0,Y0,X1,Y1,XC,YC,ID,STILL_COUNT,AREA,STATUS,MATCH)=range(MAX_PARM)
(FREE,DETECT,TRACK)=range(3)  # object STATUS options
TRACK_HEADER=('FRAME,ZC,X0,Y0,X1,Y1,XC,YC,ID,STILL_COUNT,AREA,STATUS,MATCH')

# Z tracking using Gabor filter 
SHORT_RECO=0    # regular scan for tracking object
FULL_RECO=1    # full scan for newborn object
Z_FULL_RECO_START=400
Z_FULL_RECO_STOP=800
Z_FULL_RECO_STEP_SIZE=20
Z_STEP_SIZE=20       # search step resolution 
Z_STEPS=5           # number of steps to search above and below Z value of last frame
Z_DEFAULT=600       # Where we think a newborn will start
Z_MIN=500
MIN_GABOR=1
