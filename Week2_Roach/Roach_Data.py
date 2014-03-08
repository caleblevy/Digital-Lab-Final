#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import *
from subprocess import call
import os

#os.system('scp ./RoachCommands.py root@roach:~/')
os.system('ssh root@roach ls')
#os.system('ssh root@roach rm ./RoachCommands.py')
#os.system('ssh root@roach ls')
os.system('ssh root@roach ./RoachCommands.sh')
os.system('scp 

