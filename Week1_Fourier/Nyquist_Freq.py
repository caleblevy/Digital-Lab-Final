#!/usr/bin/env python
import numpy as np
from numpy.fft import *
from matplotlib import pyplot as plt
import pylab, os
#import radiolab
ShowPlot = False

N = 256
samp_freq = 10**4 # 10 kHz sampling frequency
VppLev = 1 # 1 Volt
NPer = 5 # Number of periods (Points to display)

N_an = 1024

DataN = np.genfromtxt('data_10')

dt = 1./samp_freq
t = dt*np.arange(N)
t = t[0:(NPer+1)]

OffPhase = np.arcsin(DataN[0]/VppLev)
t_an = np.arange(N_an)*(dt*NPer/N_an)
Sig_an = np.sin(NPer*2*np.pi*t_an/t_an[N_an-1] + OffPhase)

freqAx = fftfreq(N,dt)
freqAx = fftshift(freqAx)

DataN = np.genfromtxt('data_10')

plt.figure()
#Data
plt.plot(t,DataN[0:(NPer+1)],marker='o',markerfacecolor='k',linewidth=2.5)
# Set window
pylab.ylim([-1,1])
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.grid(b=True,which='major',color=[0.4,0.4,0.4])
plt.grid(b=True,which='minor',color=[0.4,0.4,0.4])
# Schematic
plt.plot(t_an,Sig_an,linestyle='--')
# Labels
plt.xlabel('Time (seconds)',fontsize=16)
plt.ylabel('$V(t)$ (Volts)',fontsize=16)
plt.title('Signal Sampled at Nyquist Frequency',fontsize=18)

ax = plt.gca()
for I in ax.get_xticklabels() + ax.get_yticklabels():
        I.set_fontsize(14)
        
# Save
plt.savefig('Samp_is_Sig_freq.pdf')

if ShowPlot:
	plt.show()
