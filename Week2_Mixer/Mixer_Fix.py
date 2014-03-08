#!/usr/bin/env python
import numpy as np
import radiolab
import matplotlib.pyplot as plt
from numpy.fft import *

SetSRS = False
Collect = False

nu_lo = 10**5 # Local Oscillator frequency at 1 MHz
d_nu = 0.05*nu_lo # Delta nu is 5%
Pad = 20 # Safety factor

P_sig = -7.0
P_lo = -3.0

# For both

f_samp = Pad*2*nu_lo
N_samp = 2*Pad**2*nu_lo/d_nu
Period = int(round(N_samp/Pad))

dt = 1./f_samp
t = [I*dt for I in range(Period)]


# For Minus

nu_sig_minus = nu_lo - d_nu

print 'Difference of Frequencies'
print 'nu_lo=',nu_lo
print 'nu_sig_minus=',nu_sig_minus

raw_input()

if SetSRS:
	radiolab.set_srs(2,nu_lo,dbm=P_lo,off=0.0,pha=0.0)
	radiolab.set_srs(1,nu_sig_minus,dbm=P_sig,off=0.0,pha=0.0)

if Collect:
	MixMinus = radiolab.sampler(N_samp,f_samp,fileName='MixAnMinus',dual=False,low=False,integer=False,timeWarn=True)
else:
	MixMinus = np.genfromtxt('MixAnMinus')
plt.figure(0)
plt.plot(t,MixMinus[0:Period])
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.xlabel('Time (sec)')
plt.ylabel('Voltage (V)')
plt.title('Analog recording of mixer data of nu_lo-d_nu')
plt.savefig('MixerAn_Minus.pdf')

Power = abs(fft(MixMinus))
FreqAx = fftfreq(int(round(N_samp)),d=dt)
Power = fftshift(Power)
FreqAx = fftshift(FreqAx)

plt.figure(1)
plt.plot(FreqAx,Power)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Volts$^2$ ($V^2$)')
plt.title('Power spectrum of signal data in frequency space of nu_lo-d_nu')
plt.savefig('MixerAnFourrier_Minus.pdf')



# For Plus

nu_sig_plus = nu_lo + d_nu

print 'Sum of Frequencies'
print 'nu_lo=',nu_lo
print 'nu_sig_plus=',nu_sig_plus

raw_input()

if SetSRS:
	radiolab.set_srs(2,nu_lo,dbm=P_lo,off=0.0,pha=0.0)
	radiolab.set_srs(1,nu_sig_plus,dbm=P_sig,off=0.0,pha=0.0)

if Collect:
	MixPlus = radiolab.sampler(N_samp,f_samp,fileName='MixAnPlus',dual=False,low=False,integer=False,timeWarn=True)
else:
	MixPlus = np.genfromtxt('MixAnPlus')

plt.figure(2)
plt.plot(t,MixPlus[0:Period])
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.xlabel('Time (sec)')
plt.ylabel('Voltage (V)')
plt.title('Analog recording of mixer data of nu_lo+d_nu')
plt.savefig('MixerAn_Plus.pdf')

Power = abs(fft(MixPlus))
FreqAx = fftfreq(int(round(N_samp)),d=dt)
Power = fftshift(Power)
FreqAx = fftshift(FreqAx)

plt.figure(3)
plt.plot(FreqAx,Power)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Volts$^2$ ($V^2$)')
plt.title('Power spectrum of signal data in frequency space of nu_lo+d_nu')
plt.savefig('MixerAnFourrier_Plus.pdf')

plt.show()
