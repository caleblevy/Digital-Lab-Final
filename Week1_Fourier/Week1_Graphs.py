#! /usr/bin/env python
import numpy as np
from numpy.fft import *
from matplotlib import pyplot as plt
# import matplotlib.gridspec as gridspec

### Clean up this block later, and replace with more robust data loading
N = 256
samp_freq = 10**4 # 10 kHz sampling frequency
N_Pics = 10
sig_freq = samp_freq*(1./N_Pics) # (1,2,...,10)kHz
###

t_space = 1/samp_freq # Space between each sample in time

# TicMarks = 10.
# SampTicks = range(0,N,int(N/TicMarks))
t = np.arange(N)*1.0/samp_freq
tf = t[N-1]
sig = np.genfromtxt('data_low_freq')
fig = plt.figure(1)
plt.plot(t,sig)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.grid(True)
plt.xlabel('Time (seconds)')
plt.ylabel('Voltage (V)')
plt.title('Sample frequency $\ll$ signal frequency')

ax = plt.gca()
plt.text(0.7,0.5,'$f_{samp} = 10$ kHz \n' +\
                 '$f_{signal} = 10$ MHz',bbox=dict(facecolor='red', alpha=0.1),transform=ax.transAxes)
             
BigPlot = plt.figure(0)
BigPlot.subplots_adjust(left=0.125,right=0.9,bottom=0.1,top=0.9,wspace=0.5,hspace=0.2)

ax1 = plt.subplot2grid((5,4),(0,0),colspan=2)
sig = np.genfromtxt('data_1')
ax1.plot(t,sig)
plt.setp(ax1.get_xticklabels(),visible=False)

ax2 = plt.subplot2grid((5,4),(0,2),colspan=2,sharex=ax1)
sig = np.genfromtxt('data_2')
ax2.plot(t,sig)
plt.setp(ax2.get_xticklabels(),visible=False)

ax3 = plt.subplot2grid((5,4),(1,0),colspan=2,sharex=ax1)
sig = np.genfromtxt('data_3')
ax3.plot(t,sig)
plt.setp(ax3.get_xticklabels(),visible=False)

ax4 = plt.subplot2grid((5,4),(1,2),colspan=2,sharex=ax1)
sig = np.genfromtxt('data_4')
ax4.plot(t,sig)
plt.setp(ax4.get_xticklabels(),visible=False)

ax5 = plt.subplot2grid((5,4),(2,0),colspan=2,sharex=ax1)
sig = np.genfromtxt('data_5')
ax5.plot(t,sig)
plt.setp(ax5.get_xticklabels(),visible=False)

ax6 = plt.subplot2grid((5,4),(2,2),colspan=2,sharex=ax1)
sig = np.genfromtxt('data_6')
ax6.plot(t,sig)
plt.setp(ax6.get_xticklabels(),visible=False)

ax7 = plt.subplot2grid((5,4),(3,0),colspan=2,sharex=ax1)
sig = np.genfromtxt('data_7')
ax7.plot(t,sig)
plt.setp(ax7.get_xticklabels())

ax8 = plt.subplot2grid((5,4),(3,2),colspan=2,sharex=ax1)
sig = np.genfromtxt('data_8')
ax8.plot(t,sig)
plt.setp(ax8.get_xticklabels())

fftfig=plt.figure(2)
plt.plot(fft(np.genfromtxt('data_8')))
plt.show()                 
for I in range(1,N_Pics+1):
    I+1
