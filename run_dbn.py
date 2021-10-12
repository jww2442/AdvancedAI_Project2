#Utility Script for running dbn
import os
import sys
import subprocess
import numpy as np
from dbn import clear_csv, append_csv


clear_csv()
#Samples
num = 1
exception = None
flag = 0
to_write = []
while(num < 1000):
    try:
        #Action Noise
        for AN in np.arange(0, 1, 0.1):
            #Observation Noise
            for ON in np.arange(0, 1, 0.1):
                #Action Bias
                for AB in np.arange(0, 1, 0.1):
                    #No Ghosts
                    os.system('python dbn.py -s ' + str(num) + ' ' + '-a' + str(AN) + ' ' + '-o' + str(ON) + ' ' + '-b' + str(AB))
                
                    #2 Ghosts
                    os.system('python dbn.py -s ' + str(num) + ' ' + '-a' + str(AN) + ' ' + '-o' + str(ON) + ' ' + '-b' + str(AB) + ' ' + '-g' + str(2))

                    #os.system('python dbn.py -s ' + str(size) + ' ' + '-t ' + ntype + ' ' + '-n ' + str(num) + ' ' + '-a ' + str(alpha))
                    if exception:
                        break
                
                if exception:
                    break

            if exception:
                break

        if exception:
            break

    except KeyboardInterrupt:
        exception = sys.exit()

    num = num*10

    if flag:
        break

sys.exit()