#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import *
from subprocess import call
#import radiolab
import os

Collect = False
RoachGetP = False
RoachGetM = False

f_lo = 2.e6 # Local oscillator
PLev = -3.0 #DBM
d_nu = 0.5e6 # 250 kHz

f_sig_p = f_lo + d_nu
f_sig_m = f_lo - d_nu

# Plus Part

if Collect:
	radiolab.set_srs(1,freq=f_lo,dbm=PLev,off=0.0,pha=0.0)
	radiolab.set_srs(2,freq=f_sig_p,dbm=PLev,off=0.0,pha=0.0)

def PermRes():
	os.system("ssh root@roach 'chmod -f 777 ~/*'") # Reset Permissions

# I have reached a developmental milestone: I have made my first hack code
if RoachGetP:
	PermRes()
	os.system("ssh root@roach 'rm -rf ~/*'")
	os.system("scp ./Roach_Mixer.sh root@roach:~/")
	PermRes()
	os.system("ssh root@roach './Roach_Mixer.sh'")
	os.system("scp -r root@roach:~/* ./MixerP_Roach/")
	os.system("ssh root@roach 'rm -rf ~/*'")

BaseDir = os.getcwd()

N_samp = 2048 # Set by adc
f_samp = 200e6 # 200 MHz
dt = 1./f_samp
t = [I*dt for I in range(N_samp)]

PerPoints = int(1.0*f_lo*N_samp/f_samp)


FreqAx = fftfreq(N_samp,dt)
FreqAx = fftshift(FreqAx)

os.chdir('MixerP_Roach')
DataP = np.fromfile('mix_bram',dtype='>i4')/128.**2
plt.figure(0)
plt.plot(t[0:1024],DataP[0:1024],color='red') ######## Plot
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0),fontsize=16)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0),fontsize=16)
plt.xlabel('Time (sec)',fontsize=18)
plt.ylabel('Amplitude',fontsize=18)
plt.title('Waveform of '+r'$\nu_{sig}=\nu_{lo}+\Delta\nu$',fontsize=20)

plt.savefig('MixerRoachP_Phys.pdf')

plt.figure(1)
DataF_P = fftshift(abs(fft(DataP)))
Fudge = 128
plt.plot(FreqAx[N_samp/2-Fudge:N_samp/2+Fudge],DataF_P[N_samp/2-Fudge:N_samp/2+Fudge],color='brown')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0),fontsize=16)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0),fontsize=16)
plt.xlabel('Frequency (Hz)',fontsize=18)
plt.title('Power Spectrum of '+r'$\nu_{sig}=\nu_{lo}+\Delta\nu$',fontsize=20)
plt.savefig('MixerRoachP_Four.pdf')

os.chdir(BaseDir)

# Minus Part

if Collect:
        radiolab.set_srs(1,freq=f_lo,dbm=PLev,off=0.0,pha=0.0)
        radiolab.set_srs(2,freq=f_sig_m,dbm=PLev,off=0.0,pha=0.0)

if RoachGetM:
	PermRes()
        os.system("ssh root@roach 'rm -rf ~/*'")
        os.system("scp ./Roach_Mixer.sh root@roach:~/")
        PermRes()
        os.system("ssh root@roach './Roach_Mixer.sh'")
        os.system("scp -r root@roach:~/* ./MixerM_Roach/")
        os.system("ssh root@roach 'rm -rf ~/*'")

os.chdir('MixerM_Roach')
DataM = np.fromfile('mix_bram',dtype='>i4')
plt.figure(2)
plt.plot(t,DataM)
plt.savefig('MixerRoachM_Phys.pdf')

plt.figure(3)
DataF_M = fftshift(abs(fft(DataM)))
plt.plot(FreqAx,DataF_M)
plt.savefig('MixerRoachM_Four.pdf')

os.chdir(BaseDir)

#plt.show()

