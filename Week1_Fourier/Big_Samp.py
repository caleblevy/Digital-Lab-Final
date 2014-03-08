import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import *
#import DFEC


# n_samp = 10.e3


#DFEC.set_srs(1,freq=n_samp,off=0.0,pha=0.0,dbm=0.0)

#DFEC.sampler(256, n_samp, fileName='same_freq', dual=False, low=False, integer=False, timeWarn=True)

    
    
f_samp = 2.7*10.e3
n_samp = 256
sig_big_samp = np.genfromtxt('sig_big_samp')

plt.figure(0)

t = np.arange(n_samp)/f_samp
plt.plot(t,sig_big_samp)
plt.xlabel('Time (sec)',fontsize=16)
plt.ylabel('Voltage (Volts)',fontsize=16)
plt.title('Plot of waveform for'+r'$f_{samp}\ll f_{sig}$',fontsize=20)
plt.savefig('Way_Under_Samp.pdf')

sig_Four = fft(sig_big_samp)
sig_pow = abs(fftshift(sig_Four))
FreqAx = fftfreq(n_samp,1./f_samp)
FreqAx = fftshift(FreqAx)

plt.figure(1)

plt.plot(FreqAx,sig_pow,color='maroon')
plt.xlabel('Frequency (Hz)',fontsize=16)
plt.ylabel('Voltage * 256',fontsize=16)
plt.title('Plot of power spectrum for'+r'$f_{samp}\ll f_{sig}$',fontsize=20)
plt.savefig('Way_Under_Samp_Four.pdf')

#DFEC.set_srs(1,freq=10.e5,off=0.0,pha=0.0,dbm=0.0)

#DFEC.sampler(256, n_samp, fileName='sig_big_samp', dual=False, low=False, integer=False, timeWarn=True)

    
    
