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

lo_nums = [1,2,4,8]
BaseDir = os.getcwd()

# Plus Part

if Collect:
	radiolab.set_srs(1,freq=f_lo,dbm=PLev,off=0.0,pha=0.0)

def PermRes():
	os.system("ssh root@roach 'chmod -f 777 ~/*'") # Reset Permissions

# I have reached a developmental milestone: I have made my first hack code
if RoachGet:
	os.system('rm -rf lo_*')
	os.system('mkdir -p Data_Temp_lo')

	PermRes()
	os.system("ssh root@roach 'rm -rf ~/*'")
	os.system("scp Roach_lo.sh root@roach:~/")
	PermRes()
	os.system("ssh root@roach './Roach_lo.sh'")
	os.chdir('Data_Temp_lo')
	os.system("scp -r root@roach:~/* ./")
	os.system("ssh root@roach 'rm -rf ~/*'")
	
	os.system("mv lo_* ../")
	os.chdir(BaseDir)
	os.system("rm -rf Data_Temp_lo")
	

N_samp = 2048 # Set by adc
f_samp = 200e6 # 200 MHz
dt = 1./f_samp
t = [I*dt for I in range(N_samp)]

FreqAx = fftfreq(N_samp,dt)
FreqAx = fftshift(FreqAx)

FudgePhys = 64
FudgeFour = 1024

os.chdir('lo_mix_8')
lo_F = 1.0*(8./256)*f_samp
Data_Sin = np.fromfile('sin_bram','>i4')
Data_Cos = np.fromfile('cos_bram','>i4')

Zero_Freq = int(round(-13.6*10**6/(f_samp/2)*N_samp))+N_samp/2
Tone_Four = np.zeros(N_samp)
Tone_Four[Zero_Freq] = 1
Tone_Four = fftshift(Tone_Four)

Tone_Phys = ifft(Tone_Four)*1.5*10**9
Tone_re = np.real(Tone_Phys)
Tone_im = np.imag(Tone_Phys)

plt.figure(0)
plt.plot(t[0:FudgePhys],Tone_re[0:FudgePhys],linewidth=1.5,color='blue',linestyle='--')
plt.plot(t[0:FudgePhys],Tone_im[0:FudgePhys],linewidth=1.5,color='orange',linestyle='--')

plt.figure(0)
plt.plot(t[0:FudgePhys],Data_Cos[0:FudgePhys],color='blue',linewidth=1.5) # Cosine in blue
plt.plot(t[0:FudgePhys],Data_Sin[0:FudgePhys],color='orange',linewidth=1.5) # Sine in orange
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0),fontsize=16)
ax = plt.gca()
ax.set_yticklabels([])
plt.ylabel('Amplitude',fontsize=18)
plt.xlabel('Time (sec)',fontsize=18)
plt.title('Waveform of '+r'$\nu_{sig}$ mixed with onboard local oscillator',fontsize=20)
plt.savefig('lo_mix_8_Phys.pdf')

Data_Exp = Data_Cos + 1j*Data_Sin
Data_Four = fft(Data_Exp)

Data_Power = fftshift(abs(Data_Four))
plt.figure(1)
plt.plot(FreqAx[N_samp/2-FudgeFour:N_samp/2+FudgeFour],Data_Power[N_samp/2-FudgeFour:N_samp/2+FudgeFour],linewidth=1.5,color='red')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0),fontsize=16)
ax = plt.gca()
ax.set_yticklabels([])
plt.xlabel('Frequency (Hz)',fontsize=18)
plt.ylabel('Amplitude',fontsize=18)
plt.title('Power Spectrum of '+r'$\nu_{sig}$ mixed with onboard local oscillator',fontsize=18)
plt.savefig('lo_mix_8_Four.pdf')




os.chdir(BaseDir)

plt.show()


# for I in lo_nums:
#     lo_freq = I
#     lo_F = 1.0*(I/256)*f_samp
#     
#     os.chdir('lo_mix_'+str(I))
#     Data_Sin = np.fromfile('sin_bram','>i4')
#     Data_Cos = np.fromfile('cos_bram','>i4')
#     
#     plt.figure(3*I)
#     plt.plot(t,Data_Sin)
#     plt.figure(3*I+1)
#     plt.plot(t,Data_Cos)
# 
#     Data_Exp = Data_Sin + 1j*Data_Cos
#     Data_Four = fft(Data_Exp)
#     Data_Four = fftshift(abs(Data_Four))
#     plt.figure(3*I+3)
#     plt.plot(FreqAx,Data_Four)
#     
#     os.chdir(BaseDir)



#plt.show()
