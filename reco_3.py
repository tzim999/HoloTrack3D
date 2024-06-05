# -*- coding: utf-8 -*-
"""
v1 4.26.24
v2 4.29.24 added recoScanRange
 
@author: ThomasZimmerman
"""
import numpy as np
import cv2
import gaborFilter as g
import common as C

clip = lambda x, l, u: l if x < l else u if x > u else x # clip routine clip(var,min,max)

Z_STEP_SIZE=20       # search step resolution 
Z_STEPS=2           # number of steps to search above and below Z value of last frame


def recoRange(trackObj,ti,grayIM,recoFlag):
    if C.DEBUG:
        print('******** RECO ***********')
        print('START ti',ti,'trackObj[ti,C.ZC]',trackObj[ti,C.ZC]) 
    #get crop window from detector
    x0=trackObj[ti,C.X0];  x1=trackObj[ti,C.X1];
    y0=trackObj[ti,C.Y0];  y1=trackObj[ti,C.Y1];
    (ymax,xmax)=grayIM.shape
    
    #enlarge the crop window to capture fringes for better reconstruction
    xx0=clip(x0-C.W,0,xmax); xx1=clip(x1+C.W,0,xmax);  
    yy0=clip(y0-C.W,0,ymax); yy1=clip(y1+C.W,0,ymax); 
    cropIM=grayIM[yy0:yy1,xx0:xx1]
    
    # establish reconstruction locations
    if recoFlag==C.FULL_RECO: # used for newborn object
        startReco=C.Z_FULL_RECO_START
        stopReco=C.Z_FULL_RECO_STOP
        incReco=C.Z_FULL_RECO_STEP_SIZE
    else: # used for tracking existing object
        startReco=trackObj[ti,C.ZC] - C.Z_STEPS*C.Z_STEP_SIZE
        stopReco=trackObj[ti,C.ZC]  + C.Z_STEPS*C.Z_STEP_SIZE
        incReco=C.Z_STEP_SIZE
        if startReco<0:
            startReco=C.Z_MIN 
            stopReco=2*C.Z_STEPS*C.Z_STEP_SIZE
    if C.DEBUG:
        print('ti',ti,'startReco',startReco,'stopReco',stopReco,'incReco',incReco)
            
    
    # find best z using Gabor filter as focus metric (larger value means better focus)
    maxGabor=0; bestZ=C.Z_DEFAULT
    for z in range(startReco, stopReco+1, incReco): # +1 so it includes stopReco
        if z<0:     # don't allow reconstruction below zero
            z=0
        recoIM=recoFrame(cropIM,z)
        gabor=g.gaborFilter(recoIM)
        if gabor>maxGabor and gabor>C.MIN_GABOR:
            maxGabor=gabor
            bestZ=z
        if C.DEBUG:
            print('ti',ti,'z',z,'Gabor:',int(gabor),'bestZ',bestZ)    
    trackObj[ti,C.ZC]=bestZ                 # save best focus Z
    if C.DEBUG:
        print('FINISH ti',ti,'trackObj[ti,C.ZC]',trackObj[ti,C.ZC])    

    recoIM=recoFrame(cropIM,bestZ)
    return(recoIM,bestZ,maxGabor)

def recoFrame(cropIM, z):
    #make even coordinates
    (yRez,xRez)=cropIM.shape
    if (xRez%2)==1:
        xRez-=1
    if (yRez%2)==1:
        yRez-=1
    cropIM=cropIM[0:yRez,0:xRez]
    complex = propagate(np.sqrt(cropIM), C.wvlen, z*C.zScale, C.dxy)	 #calculate wavefront at z
    amp = np.abs(complex)**2          # output is the complex field, still need to compute intensity via abs(res)**2
    ampInt = amp.astype('uint8')
    return(ampInt)

def propagate(input_img, wvlen, zdist, dxy):
    M, N = input_img.shape # get image size, rows M, columns N, they must be even numbers!

    # prepare grid in frequency space with origin at 0,0
    _x1 = np.arange(0,N/2)
    _x2 = np.arange(N/2,0,-1)
    _y1 = np.arange(0,M/2)
    _y2 = np.arange(M/2,0,-1)
    _x  = np.concatenate([_x1, _x2])
    _y  = np.concatenate([_y1, _y2])
    x, y  = np.meshgrid(_x, _y)
    kx,ky = x / (dxy * N), y / (dxy * M)
    kxy2  = (kx * kx) + (ky * ky)

    # compute FT at z=0
    E0 = np.fft.fft2(np.fft.fftshift(input_img))

    # compute phase aberration
    _ph_abbr   = np.exp(-1j * np.pi * wvlen * zdist * kxy2)
    output_img = np.fft.ifftshift(np.fft.ifft2(E0 * _ph_abbr))
    return output_img