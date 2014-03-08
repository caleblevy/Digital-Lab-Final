#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import *
from subprocess import call
import radiolab
import os

Collect = False
RoachGet = True

f_lo = 2.e6 # Local oscillator
PLev = -3.0 #DBM
d_nu = 0.25*f_lo

f_sig_p = f_lo + d_nu
f_sig_m = f_lo - d_nu

# Plus Part

if Collect:
	radiolab.set_srs(1,freq=f_lo,dbm=PLev,off=0.0,pha=0.0)
	radiolab.set_srs(2,freq=f_sig_p,dbm=PLev,off=0.0,pha=0.0)

def PermRes():
	os.system("ssh root@roach 'chmod -f 777 ~/*'") # Reset Permissions

# I have reached a developmental milestone: I have made my first hack code
if RoachGet:
	PermRes()
	os.system("ssh root@roach 'rm -rf ~/*'")
	os.system("scp ./Roach_Mixer.sh root@roach:~/")
	PermRes()
	os.system("ssh root@roach './Roach_Mixer.sh'")
	os.system("scp -r root@roach:~/* ./MixerP/")
	os.system("ssh root@roach 'rm -rf ~/*'")

BaseDir = os.getcwd()

N_samp = 2048 # Set by adc
f_samp = 200e6 # 200 MHz
dt = 1./f_samp
t = [I*dt for I in range(N_samp)]

FreqAx = fftfreq(N_samp,dt)
FreqAx = fftshift(FreqAx)

os.chdir('MixerP')
Data = np.fromfile('adc_bram',dtype='>i4',sep=' ')
plt.figure(0)
plt.plot(t,Data)

plt.figure(1)
DataF = fftshift(abs(fft(Data)))
plt.plot(FreqAx,DataF)
plt.show()
