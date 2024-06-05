# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 12:13:29 2024

@author: ThomasZimmerman
"""
import common as C # constants used by all programs
import numpy as np
import math
import reco_3 as R # for full reco of newborn
import cv2

nextID=1 # start with ID=1
def findID(detectObj,trackObj,grayIM): 
    global nextID
    
    if C.DEBUG:
        print('******** TRACK *********')
    
    #create index list active detected and tracking objects for matching, free list for newborns 
    detectList=[]; trackList=[]; freeList=[]
    for di in range(C.MAX_DETECT):      
        if detectObj[di,C.STATUS]==C.DETECT:
            detectList.append(di)
    for ti in range(C.MAX_TRACK):    
        if trackObj[ti,C.STATUS]==C.TRACK:
            trackList.append(ti)
        if trackObj[ti,C.STATUS]==C.FREE:
            freeList.append(ti)
    
    # match closest detect object with track object        
    d=np.zeros((0))     # array holding distance between all combinations of tracked and detected objects
    detectIndex=[]; trackIndex=[];  # keep track of obj locations
    if C.DEBUG:
        print('detectList',detectList)
        print('trackList',trackList)
        print('Z trackList',trackObj[trackList,C.ZC])
    for di in detectList:       # detected objects
        for ti in trackList:    # tracked objects
            detectIndex.append(di);  trackIndex.append(ti)  # save the index so we can find them with the minimum distance array (dsort)
            dx=detectObj[di,C.XC]-trackObj[ti,C.XC]         # x distance between track and detect object
            dy=detectObj[di,C.YC]-trackObj[ti,C.YC]         # y distance between track and detect object
            distance=int(math.sqrt(dx*dx+dy*dy))            # abs distance between track and detect object
            d=np.append(d,distance)           # append distance to list
            if C.DEBUG:
                print(di,'append distance:',int(distance),'d',d)
    
    #match track obj to closest detect obj
    detectObj[:,C.MATCH]=0; trackObj[:,C.MATCH]=0;          # initialize all tracked and detected objects as not matched
    dsort=np.argsort(d)    # get index of distances sorted in ascending order (small to large)
    if C.DEBUG:
        print('dsort',dsort)
        print('detectIndex',detectIndex)  
        print('trackIndex',trackIndex)  
    
    for i in range(len(d)):
        index=dsort[i]                                      # get the index from list of ascending distances 
        di=detectIndex[index]; ti=trackIndex[index]         # index into the objects in detectList and trackList
        if detectObj[di,C.MATCH]==0 and trackObj[ti,C.MATCH]==0:  # only match unmatched objects 
            detectObj[di,C.MATCH]=1         # indicate detect and track objects are matched so they are only matched once
            trackObj[ti,C.MATCH]=1          
            updateTrackLocation(di,ti,detectObj,trackObj)       # transfer detect obj location to track obj
            trackObj[ti,C.STILL_COUNT]=0                # indicate tracked object is moving
            if C.DEBUG:
                print('ti',ti,'match ID',trackObj[ti,C.ID],'trackObj[ti,C.ZC]',trackObj[ti,C.ZC])
    
    # check if any unmatched track obj are dead, and if so set them free to be assigned in the future
    for ti in trackIndex: # tracked obj
        if trackObj[ti,C.MATCH]==0:
            trackObj[ti,C.STILL_COUNT]+=1       # indicate track obj location is not detected (moving) during this frame
            if trackObj[ti,C.STILL_COUNT]>C.MAX_STILL:
                trackObj[ti,C.STATUS]=C.FREE            # not moving for a while so set it free to track a new detected object
                #cv2.destroyAllWindows()  # kill all windows because sometimes little window stick around even when not updated
    # if unmatched detect obj, assign to FREE track object (if available) with new ID
    #print('detectObj',detectObj[detectList,C.MATCH])
    for di in detectList:
        if detectObj[di,C.MATCH]==0:
            # if C.DEBUG:
            #     print('di',di,'prepop freeList',freeList)
            if len(freeList)>0:
                tiFree=freeList[0]
                freeList.pop(0) # remove ti from free list
                # if C.DEBUG:
                #     print('di',di,'postpop freeList',freeList)
                (detectObj,trackObj)=updateTrackLocation(di,tiFree,detectObj,trackObj)
                trackObj[tiFree,C.STILL_COUNT]=0                # indicate tracked object is moving
                trackObj[tiFree,C.STATUS]=C.TRACK               # start tracking object
                (recoIM,bestZ,maxGabor)=R.recoRange(trackObj,tiFree,grayIM,C.FULL_RECO)  # assign new object Z from full Z stack reco
                trackObj[tiFree,C.ZC]=bestZ
                trackObj[tiFree,C.ID]=nextID
                if C.DEBUG:
                    print('tiFree',tiFree,'z',trackObj[tiFree,C.ZC],'id',trackObj[tiFree,C.ID])
                nextID+=1 # prepare for next newborn so we don't reuse ID's
    if C.DEBUG:
        print('detectList',len(detectList),'trackList',len(trackList),'nextID',nextID)
        print('Z trackList',trackObj[trackList,C.ZC])
    return(detectObj,trackObj)

def updateTrackLocation(di,ti,detectObj,trackObj):
    trackObj[ti,C.AREA]=detectObj[di,C.AREA]    # tracked obj inherets detect area
    trackObj[ti,C.XC]=detectObj[di,C.XC]        # tracked obj inherets detect location
    trackObj[ti,C.YC]=detectObj[di,C.YC]    
    trackObj[ti,C.X0]=detectObj[di,C.X0]        
    trackObj[ti,C.X1]=detectObj[di,C.X1]        
    trackObj[ti,C.Y0]=detectObj[di,C.Y0]        
    trackObj[ti,C.Y1]=detectObj[di,C.Y1]   
    return(detectObj,trackObj)