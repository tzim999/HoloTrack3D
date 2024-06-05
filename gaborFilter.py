# Run a stack of images through a 0 and 90 deg Gabor Filter
# Select the image with the highest Gabor value
# Tom Zimmerman, IBM Research-Almaden, CCC, 5.1.23

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel
#from os.path import isfile, join
#from os import listdir
#import cv2
#from matplotlib import pyplot as plt
import math


def convolve(image, kernel):
    feat = np.zeros((len(kernel)), dtype=np.double)
    filtered = ndi.convolve(image, kernel, mode='wrap')
    feat = filtered.var()
    return feat

################## MAIN ##########################

# prepare filter kernels
theta=0
angle = theta / 4. * np.pi
sigma=1 # 4,9,12
frequency=0.35 #0.125, 0.25,0.35,0.50
kernel0 = np.real(gabor_kernel(frequency, theta=angle,sigma_x=sigma, sigma_y=sigma))
theta=90
angle = theta / 4. * np.pi
kernel90 = np.real(gabor_kernel(frequency, theta=angle,sigma_x=sigma, sigma_y=sigma))
    
def gaborFilter(im):
    gaborValue0=convolve(im, kernel0)
    gaborValue90=convolve(im, kernel90)
    gaborValue=math.sqrt(gaborValue0**2+gaborValue90**2)
    return(gaborValue)

