# Detect objects 
import numpy as np
import cv2
import keyboard as K
import reco_3 as R
import common as C
import track_3 as T
np.set_printoptions(suppress=True) # supress scientific notation of printed values

#
# Thomas Zimmerman IBM Research-Almaden, Center for Cellular Construction (https://ccc.ucsf.edu/) 
# This work is funded by the National Science Foundation (NSF) grant No. DBI-1548297 
# Disclaimer:  Any opinions, findings and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation.

trackFile='m2a_Track.csv'
videoFile='m2a.mp4'

detectObj=np.zeros((C.MAX_DETECT,C.MAX_PARM),dtype=int)      # detected objects, changes each frame
trackObj=np.zeros((C.MAX_TRACK,C.MAX_PARM),dtype=int)    # tracked objects, persists across frames, ID added by track code
trackLog=np.zeros((0,C.MAX_PARM),dtype=int)

gaborObj=np.zeros((C.MAX_TRACK),dtype=int)
def doReco(trackObj,grayIM):
    for ti in range(C.MAX_TRACK):
        if trackObj[ti,C.STATUS]==C.TRACK:
            (recoIM,bestZ,maxGabor)=R.recoRange(trackObj,ti,grayIM,C.SHORT_RECO)
            (r,c)=recoIM.shape
            bigRecoIM=cv2.resize(recoIM, (C.IMAGE_SCALE*c,C.IMAGE_SCALE*r)) # enlarge to see reco better
            objID=trackObj[ti,C.ID] # display obj number on reco image window
            cv2.imshow(str(objID),bigRecoIM)
            gaborObj[ti]=maxGabor
      
def updateTrackLog(frameCount,trackObj):
    global trackLog # make global so we can save data into trackLog
    trackVector=np.zeros((1,C.MAX_PARM),dtype=int)
    trackObj[:,C.FRAME]=frameCount
    ti=0    # track array index
    for ti in range(len(trackObj)):
        if trackObj[ti,C.ID]!=0 and trackObj[ti,C.ID]==C.TRACK: # ID=0 means no tracking for that track obj so don't save row
            trackVector[0,:]=trackObj[ti]
            trackLog=np.append(trackLog,trackVector,axis=0)
            
def detect(videoFile):
    global trackObj,detectObj
    
    frameCount=1
    cap = cv2.VideoCapture(videoFile)
    while(cap.isOpened()):  # start frame capture
        #key=cv2.waitKey(10)& 0xFF # value is delay in milliseconds to slow down video, must be > 0
        #if key==ord('q'):
            #break
        (update,run,keyList)=K.processKey()
        if run==0:
            break  # end program
        (thresh,blur,minArea,maxArea)=keyList
        minArea*=10; maxArea*=10
        #print(keyList)
        if update:
            print('thresh:',thresh,'blur:',blur,'minArea:',minArea,'maxArea:',maxArea)
        
        # get image
        ret, colorIM = cap.read()
        rectIM=np.copy(colorIM) # make copy that can be marked up with rectangles
        if not ret: # check to make sure there was a frame to read
            print('End of video detected, so finish')
            break
        
        # do image processing
        grayIM = cv2.cvtColor(colorIM, cv2.COLOR_BGR2GRAY)     # convert color to grayscale image
        blurIM=cv2.medianBlur(grayIM,(2*blur)+1)                 # blur image to fill in holes to make solid object
        ret,threshIM = cv2.threshold(blurIM,thresh,255,cv2.THRESH_BINARY_INV) # threshold image to make pixels 0 or 255
        contourList, hierarchy = cv2.findContours(threshIM, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # all countour points, uses more memory
        
        detectObj[:,C.STATUS]=C.FREE     # prepare for new object detection
        objCount=0
        for objContour in contourList:
            area = cv2.contourArea(objContour)
            if area>minArea and area<maxArea and objCount<C.MAX_DETECT:    # only process objects of good size that don't touch image edge
                # Get bounding box for ROI
                PO = cv2.boundingRect(objContour)
                x0=PO[0]; y0=PO[1]; x1=x0+PO[2]; y1=y0+PO[3]; xc=x0+(x0+x1)/2; yc=y0+(y0+y1)/2;
                detectObj[objCount,C.X0]=x0; detectObj[objCount,C.Y0]=y0; 
                detectObj[objCount,C.X1]=x1; detectObj[objCount,C.Y1]=y1; 
                detectObj[objCount,C.XC]=xc; detectObj[objCount,C.YC]=yc; 
                detectObj[objCount,C.AREA]=area; detectObj[objCount,C.STILL_COUNT]=0; 
                detectObj[objCount,C.STATUS]=C.DETECT
                rectIM=cv2.rectangle(colorIM, (x0,y0), (x1,y1), (0,255,0), 4)  
                objCount+=1
        
        if (C.TRACK_RECO_ENABLE):  # disable to speed up detection to adjust parameters
            (detectObj,trackObj)=T.findID(detectObj,trackObj,grayIM)           
            doReco(trackObj,grayIM)
        
        
        cv2.imshow('threshIM', cv2.resize(threshIM, C.DISPLAY_REZ))      # display reduced image
        cv2.imshow('rectIM', cv2.resize(rectIM, C.DISPLAY_REZ))         # display reduced image
        #cv2.imshow('blurIM', cv2.resize(blurIM, C.DISPLAY_REZ))        # display reduced image
        
        if C.DEBUG:
            print() # new line for new frame
            print('************* DETECT ***********')
            print('Detect',detectObj[:,C.STATUS])
            print('Z',trackObj[:,C.ZC])
        
        if C.DEBUG_Z:
            print('frame',frameCount,'trackObj[:,C.ZC]',trackObj[:,C.ZC])
            #print('\tgaborObj[:]',gaborObj[:])
        
        updateTrackLog(frameCount,trackObj)
        frameCount+=1
    print('thresh:',thresh,'blur:',blur,'minArea:',minArea,'maxArea:',maxArea)
    cap.release()
    cv2.destroyAllWindows()

def startDetect():
    #XC,YC,ZC,ID,STILL_COUNT,AREA,STATUS
    detectObj[:,C.STATUS]=C.FREE
    trackObj[:,C.STATUS]=C.FREE
        
########## MAIN ###########
print('Processing',videoFile,' Press "q" key to quit')
startDetect()
detect(videoFile)
print("Saving tracking file: ",trackFile)
np.savetxt(trackFile,trackLog,header=C.TRACK_HEADER,fmt='%i',delimiter=',') # saves numpy array as a csv file    
            

