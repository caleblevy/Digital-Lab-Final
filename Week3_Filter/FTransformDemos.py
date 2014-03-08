#!/usr/bin/env python
import numpy as np 
from numpy.fft import *
from matplotlib import pyplot as plt
#import radiolab
import os
from RoachNum import *

NN = 8
Window = 5 # Centered window

Fir = np.zeros(NN)
Fir[NN/2+1:NN/2+1+(Window-1)/2] = 1.0
Fir[NN/2-(Window-1)/2:NN/2+1] = 1.0

FirShift = fftshift(Fir)
FirPhys = ifft(FirShift)

FirCoeffs = fftshift(FirPhys)
FirCoeffs = np.real(FirCoeffs)

# HexCoeffs = [None]*N
# 
# for I in range(N):
#     HexCoeffs[I] = Fix_18_17(FirCoeffs[I])
# print FirCoeffs 
#     
# for I in range(N):
#     print To_Binary(FirCoeffs[I],18,0)
# 
# print HexCoeffs    
