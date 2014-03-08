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

## Make the main physical figure
figPhys,axesPhys = plt.subplots(ncols=2,nrows=N_Pics/2)
## Make the Last appended grid
gs = gridspec.GridSpec(N_Pics/2,3,width_ratios=[1,2,1])
axLast = plt.subplot(gs[N_Pics/2-1,1])
figPhys.subplots_adjust(hspace=0.3)
## Make the framing axes, and set to invisible
axBig = figPhys.add_subplot(111)
# Turn off all the stuff but labels
axBig.spines['top'].set_color('none')
axBig.spines['bottom'].set_color('none')
axBig.spines['left'].set_color('none')
axBig.spines['right'].set_color('none')
axBig.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
# BigASS XLABEL
axBig.set_xlabel('Time (seconds)',fontsize=16,labelpad=20)
axBig.set_ylabel('Voltage (Volts)',fontsize=16,labelpad=20)



count = 0
for I in axesPhys[0:N_Pics/2-1]:
    count += 1
    J = True
    for ax in I:
        if J:
            freq_num = count
        else:
            freq_num = 10-count
            ax.set_yticklabels([])
            
        sig = np.genfromtxt('data_'+str(freq_num))
        ax.set_title(r'$\nu_{sig}=%s\nu_{samp}$'%('0.'+str(freq_num)))
        
        sig = sig[0:N_PlotPoints]
        
        if not count == N_Pics/2-1:
            ax.set_xticklabels([])
        else:
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            
            
            
        figPhys.add_subplot(ax)
        if J and (not count == N/2-2):
            ax.plot(t*1e3/1.1,sig,'g',marker='o',markerfacecolor='k')
        else:
            ax.plot(t,sig,'g',marker='o',markerfacecolor='k')
        
        
        
        pylab.ylim([-1.0,1.0])
        J = False
# Semilast scientific

        
plt.suptitle('Sampling Error in the Time Domain',fontsize=20)
for ax in axesPhys[N_Pics/2-1]:
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    


# Last Grid
figPhys.add_subplot(axLast)
sig = np.genfromtxt('data_'+str(N_Pics/2))
sig = sig[0:N_PlotPoints]
axLast.plot(t,sig,'g',marker='o',markerfacecolor='k')
axLast.set_title(r'$\nu_{sig}=\nu_{Ny}$')
axLast.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

fig = plt.gcf()
DefaultSize = fig.get_size_inches()
fig.set_size_inches((DefaultSize[0]/1.25,DefaultSize[1]*1.75))
pylab.ylim([-1.0,1.0])


fig.savefig('Time_Dom_Samp_Set.pdf')





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


# plt.show()
# plt.show()
    
    
    

    
