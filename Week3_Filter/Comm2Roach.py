#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import *
from subprocess import call
import radiolab
import os

Collect = False
RoachGet = False

f_lo = 1.e6 # Local oscillator
P_lo = -10.0 #DBM

if Collect:
	radiolab.set_srs(1,freq=f_lo,dbm=P_lo,off=0.0,pha=0.0)

def PermRes():
	os.system("ssh root@roach 'chmod -f 777 ~/*'") # Reset Permissions

# I have reached a developmental milestone: I have made my first hack code
if RoachGet:
	PermRes()
	os.system("ssh root@roach 'rm -rf ~/*'")
	os.system("scp ./Roach_Simple.sh root@roach:~/")
	PermRes()
	os.system("ssh root@roach './Roach_Simple.sh'")
	os.system("scp -r root@roach:~/* ./DataFolds/")
	os.system("ssh root@roach 'rm -rf ~/*'")

os.chdir('DataFolds')
BaseDir = os.getcwd()

N_samp = 2048 # Set by adc
f_samp = 200e6 # 200 MHz
dt = 1./f_samp
t = [I*dt for I in range(N_samp)]

FreqAx = fftfreq(N_samp,dt)
FreqAx = fftshift(FreqAx)


# Step through the folders
os.chdir('adc_SampleSig')
Data = np.fromfile('adc_bram','>i4')
plt.figure(0)
plt.plot(t,Data)

plt.figure(1)
DataF = fft(Data)
PowerF = fftshift(abs(DataF))

plt.plot(FreqAx,PowerF)

# Check the Mixed data
os.chdir(BaseDir)
os.chdir('adc_lo_freq_8')

Data = np.fromfile('adc_bram','>i4')
plt.figure(2)
plt.plot(t,Data)

plt.figure(3)
DataF = fft(Data)
PowerF = fftshift(abs(DataF))

plt.plot(FreqAx,PowerF)

plt.show()

'''
Data_Cos = np.fromfile('cos_bram','>i4')
Data_Sin = np.fromfile('sin_bram','>i4')
plt.figure(2)
plt.plot(t,Data_Cos)
plt.plot(t,Data_Sin)

Data_Sig = Data_Cos + 1j*Data_Sin
DataPower = fft(Data_Sig)
DataPower = fftshift(abs(DataPower))
plt.figure(3)
plt.plot(FreqAx,DataPower)

plt.show()
'''
