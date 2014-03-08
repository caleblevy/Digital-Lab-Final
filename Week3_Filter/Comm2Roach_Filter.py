#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import *
from subprocess import call
#import radiolab
import os

Collect = False
RoachGet = False

f_sig = 20.e6 # Local oscillator
PLev = -3.0 #DBM

lo_nums = [8]
BaseDir = os.getcwd()

NN = 8
Window = 5 # Centered window

Fir = np.zeros(NN)
Fir[NN/2+1:NN/2+1+(Window-1)/2] = 1.0
Fir[NN/2-(Window-1)/2:NN/2+1] = 1.0

FirShift = fftshift(Fir)
FirPhys = ifft(FirShift)

FirCoeffs = fftshift(FirPhys)
FirCoeffs = np.real(FirCoeffs)

PadN = 1024
FirPad = np.zeros(2*PadN)
for I in range(NN):
    # print PadN-NN/2+I,FirCoeffs[I],I
    
    FirPad[PadN-NN/2+I] = FirCoeffs[I]

PadShift = fftshift(FirPad)
PadCoeffs = fft(PadShift)
PadCoeffs = fftshift(PadCoeffs)


# Plus Part

if Collect:
    radiolab.set_srs(1,freq=f_lo,dbm=PLev,off=0.0,pha=0.0)

def PermRes():
    os.system("ssh root@roach 'chmod -f 777 ~/*'") # Reset Permissions

# I have reached a developmental milestone: I have made my first hack code
if RoachGet:
    os.system('rm -rf lo_*')
    os.system('mkdir -p Data_Temp_Filter')

    PermRes()
    os.system("ssh root@roach 'rm -rf ~/*'")
    os.system("scp Roach_Filter.sh root@roach:~/")
    PermRes()
    os.system("ssh root@roach './Roach_Filter.sh'")
    os.chdir('Data_Temp_Filter')
    os.system("scp -r root@roach:~/* ./")
    os.system("ssh root@roach 'rm -rf ~/*'")
    
    os.system("mv lo_* ../")
    os.chdir(BaseDir)
    os.system("rm -rf Data_Temp_Filter")
    

N_samp = 2048 # Set by adc
f_samp = 200e6 # 200 MHz
dt = 1./f_samp
t = [I*dt for I in range(N_samp)]

FreqAx = fftfreq(N_samp,dt)
FreqAx = fftshift(FreqAx)

count = 0

for I in lo_nums:
    lo_freq = I
    lo_F = 1.0*(I/256)*f_samp 
    
    os.chdir('lo_Filter_'+str(I))
    Data_Sin = np.fromfile('ddc_real_bram','>i4')
    Data_Cos = np.fromfile('ddc_imag_bram','>i4')
    
    plt.figure(count+1)
    plt.plot(t,Data_Sin,'red')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0),fontsize=16)
    ax = plt.gca()
    ax.set_yticklabels([])
    plt.xlabel('Time (sec)',fontsize=18)
    plt.title('Waveform of filtered noise',fontsize=20)
    # plt.ylabel('Amplitude',fontsize=18)
    plt.savefig('Filtered_Noise_Phys.pdf')
    

    Data_Exp = Data_Sin + 1j*Data_Cos
    Data_Four = fft(Data_Exp)
    Data_Four = fftshift(abs(Data_Four))
    plt.figure(count+3)
    plt.plot(FreqAx,Data_Four,color='cyan')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0),fontsize=16)
    ax = plt.gca()
    ax.set_yticklabels([])
    lev = 1.2*10**7
    linlin = np.array([lev]*N_samp)
    plt.plot(FreqAx,linlin,linewidth=3,linestyle='--',color='orange')
    plt.plot(FreqAx,linlin*1./2,linewidth=3,linestyle='--')
    plt.xlabel('Frequency (Hz)',fontsize=18)
    plt.title('Power Spectrum of Filtered Noise',fontsize=20)
    Freq2 = fftshift(fftfreq(NN,1./f_samp))
    plt.plot(Freq2,abs(Fir*lev),color='blue',linewidth=2)
    
    Freq3 = fftshift(fftfreq(2*PadN,1./f_samp))
    plt.plot(Freq3,abs(PadCoeffs)*lev,color='green',linewidth=2)
    plt.ylabel('Amplitude',fontsize=18)
    print PadCoeffs   
    
    plt.savefig('Filtered_Noise_Four.pdf')
    plt.figure(10000)
    Data_Sin_leo = np.fromfile('ddc_real_bram_leo','>i4')
    Data_Cos_leo = np.fromfile('ddc_imag_bram_leo','>i4')
    plt.plot(t,Data_Sin_leo,color='blue')
    plt.xlabel('Time (sec)',fontsize=18)
    plt.ylabel('Amplitude',fontsize=18)
    plt.title('Waveform of unfiltered noise',fontsize=20)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0),fontsize=16)
    ax = plt.gca()
    ax.set_yticklabels([])
    plt.savefig('Leo_Random.pdf')
    
    count += 3
    
    


plt.show()
