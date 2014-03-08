#! /usr/bin/env python
import numpy as np
# import radiolab
import matplotlib.pyplot as plt
from numpy.fft import *

Collect = False

nu_lo = 10**5 # Local Oscillator frequency at 10 kHz
d_nu = 0.05*nu_lo # Delta nu is 5%
Pad = 20. # Safety factor


nu_sig = nu_lo + d_nu 
PLev = 0.0 # Default power levels in dBm
if Collect:
	radiolab.set_srs(2,nu_lo,dbm=PLev,off=0.0,pha=0.0)
	radiolab.set_srs(1,nu_sig,dbm=PLev,off=0.0,pha=0.0) # Set to defaults, with additive delta nu this time

f_samp = Pad*2*nu_lo # Five times Nyquist
N_samp = 2*Pad**2*nu_lo/d_nu # Five minimum sampling periods

N_samp = int(round(N_samp))
if Collect:
	MixAnData = radiolab.sampler(N_samp,f_samp,fileName='Mixer_Analogue',dual=False,low=False,integer=False,timeWarn=True)
else:
	MixAnData = np.genfromtxt('Mixer_Analogue')


Period = int(round(N_samp/Pad))
dt = 1/f_samp
t = [I*dt for I in range(Period)]

plt.plot(t,MixAnData[0:Period])
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.xlabel('Time (sec)',fontsize=18)
plt.ylabel('Voltage (V)',fontsize=18)
plt.title('Analog recording of saturated data in physical space',fontsize=20)
plt.savefig('MixerAn.pdf')

Power = abs(fft(MixAnData))
FreqAx = fftfreq(N_samp,dt)

Power = fftshift(Power)
FreqAx = fftshift(FreqAx)

plt.figure()
plt.plot(FreqAx,Power,color='orange')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.xlabel('Frequency (Hz)',fontsize=18)
plt.ylabel('Voltage*16,000',fontsize=18)
plt.title('Power spectrum of saturated data in frequency space',fontsize=18)
plt.savefig('MixerAnFourrier.pdf')

