#! /usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt
from numpy.fft import *
import os,pylab
import matplotlib.gridspec as gridspec
#import radiolab

DoLoop = False

N = 256
samp_freq = 10**4 # 10 kHz sampling frequency
N_Pics = 10
# sig_freq = samp_freq*(1./N_Pics) # (1,2,...,10)kHz
sig_freq = samp_freq*(1./N_Pics)*np.arange(1,N_Pics+1)

if DoLoop:
	for f in range(1,N_Pics+1):
		# radiolab.set_srs(1,freq=1.0*f*sig_freq,off=0.0,pha=0.0,vpp=1.0)
		radiolab.set_srs(1,freq=sig_freq[f-1],off=0.0,pha=0.0,vpp=1.0)
		radiolab.sampler(N,samp_freq,fileName='data_'+str(f),dual=False,low=False,integer=False,timeWarn=True)
	
	radiolab.set_srs(1,freq=10**7,off=0.0,pha=0.0,vpp=1.0)
	radiolab.sampler(N,samp_freq,fileName='data_low_freq',dual=False,low=False,integer=False,timeWarn=True)

dt = 1./samp_freq
t = dt*np.arange(N)
N_PlotPoints = int(round(N*sig_freq[0]/samp_freq/2))
t = t[0:N_PlotPoints]
freqAx = fftfreq(N,dt)
freqAx = fftshift(freqAx)



figFour,axesFour = plt.subplots(ncols=2,nrows=N_Pics/2-1)
count = 0
for I in axesFour[0:N_Pics/2-1]:
	count += 1
	J = False
	for ax in I:
		if J:
			Four = np.genfromtxt('data_'+str(count))
		else:
			Four = np.genfromtxt('data_'+str(10-count))

		Four = abs(fft(Four))
		Four = fftshift(Four)

		if not count == N_Pics/2-1:
			ax.set_xticklabels([])
		figFour.add_subplot(ax)
		ax.plot(freqAx,Four)
		J = True


#plt.show()
# plt.show()
    
    
    

    
