#!/usr/bin/env python
import numpy as np
#import radiolab
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

FreqPer = int(round((1.5*2.0*nu_lo/f_samp)*N_samp))
NN_samp = int(N_samp)

dt = 1./f_samp
t = [I*dt for I in range(Period)]


FreqAx = fftfreq(int(round(N_samp)),d=dt)
FreqAx = fftshift(FreqAx)


FreqP_Line = int(round((2.0*nu_lo/f_samp)*N_samp))
FreqP_Pos = np.array([FreqAx[N_samp/2+FreqP_Line]]*700)
FreqP_Neg = np.array([FreqAx[N_samp/2-FreqP_Line]]*700)
LLine = np.arange(700)


# For Minus

nu_sig_minus = nu_lo - d_nu

print 'Difference of Frequencies'
print 'nu_lo=',nu_lo
print 'nu_sig_minus=',nu_sig_minus


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
plt.xlabel('Time (sec)',fontsize=18)
plt.ylabel('Voltage (V)',fontsize=18)
plt.title('Analog recording of mixer data of '+r'$\nu_{lo}-\Delta\nu$',fontsize=20)
plt.savefig('MixerAn_Minus.pdf')

Power = abs(fft(MixMinus))
FreqAx = fftfreq(int(round(N_samp)),d=dt)
Power = fftshift(Power)
FreqAx = fftshift(FreqAx)
FreqAx = FreqAx

plt.figure(1)
plt.plot(FreqAx[NN_samp/2-FreqPer:NN_samp/2+FreqPer],Power[NN_samp/2-FreqPer:NN_samp/2+FreqPer],color='purple',linewidth=2.5)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0),fontsize=16)
plt.ticklabel_format(axis='y',fontsize=16)
plt.xlabel('Frequency (Hz)',fontsize=18)
plt.ylabel('Voltage*256',fontsize=18)
plt.title('Power spectrum of signal data in frequency space of '+r'$\nu_{lo}-\Delta\nu$',fontsize=18)
plt.plot(FreqP_Pos,LLine,color='blue',linestyle='--')
plt.plot(FreqP_Neg,LLine,color='blue',linestyle='--')
plt.savefig('MixerAnFourrier_Minus.pdf')



# For Plus

nu_sig_plus = nu_lo + d_nu

print 'Sum of Frequencies'
print 'nu_lo=',nu_lo
print 'nu_sig_plus=',nu_sig_plus


if SetSRS:
	radiolab.set_srs(2,nu_lo,dbm=P_lo,off=0.0,pha=0.0)
	radiolab.set_srs(1,nu_sig_plus,dbm=P_sig,off=0.0,pha=0.0)

if Collect:
	MixPlus = radiolab.sampler(N_samp,f_samp,fileName='MixAnPlus',dual=False,low=False,integer=False,timeWarn=True)
else:
	MixPlus = np.genfromtxt('MixAnPlus')

plt.figure(2)
plt.plot(t,MixPlus[0:Period],color='indigo')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0),fontsize=16)
plt.ticklabel_format(axis='y',fontsize=16)
plt.xlabel('Time (sec)',fontsize=18)
plt.ylabel('Voltage (V)',fontsize=18)
plt.title('Analog recording of mixer data of '+r'$\nu_{lo}+\Delta\nu$',fontsize=20)
plt.savefig('MixerAn_Plus.pdf')

Power = abs(fft(MixPlus))
FreqAx = fftfreq(int(round(N_samp)),d=dt)
Power = fftshift(Power)
FreqAx = fftshift(FreqAx)

plt.figure(3)
plt.plot(FreqAx[NN_samp/2-FreqPer:NN_samp/2+FreqPer],Power[NN_samp/2-FreqPer:NN_samp/2+FreqPer],color='purple',linewidth=2.5)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0),fontsize=16)
plt.ticklabel_format(axis='y',fontsize=16)
plt.xlabel('Frequency (Hz)',fontsize=18)
#plt.ylabel('Voltage*256',fontsize=18)
plt.title('Power spectrum of signal data in frequency space of '+r'$\nu_{lo}+\Delta\nu$',fontsize=18)
plt.plot(FreqP_Pos,LLine,color='blue',linestyle='--')
plt.plot(FreqP_Neg,LLine,color='blue',linestyle='--')
plt.savefig('MixerAnFourrier_Plus.pdf')


Filter_Copy = fft(MixPlus)
low_Point = 2*d_nu
Freq_Cmp = fftshift(FreqAx)
for I in range(int(N_samp)):
	if abs(Freq_Cmp[I]) > low_Point:
		Filter_Copy[I] = 0

plt.figure(4)
plt.plot(fftshift(abs(Filter_Copy)))

Filter_Phys = np.real(ifft(Filter_Copy))
plt.figure(5)
plt.plot(t,Filter_Phys[0:Period])
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0),fontsize=16)
plt.ticklabel_format(axis='y',fontsize=16)
plt.xlabel('Time (sec)',fontsize=18)
plt.title('Filtered Output of Heterodyne Mixer',fontsize=20)
plt.savefig('Filter_Plus.pdf')

#plt.show()
