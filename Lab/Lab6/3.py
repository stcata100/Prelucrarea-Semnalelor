import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


N = 2
# vectori ce contin coeficientii polinoamelor
P_1 = np.random.randint(0, 20, size=N)
P_2 = np.random.randint(0, 20, size=N)

Product = np.convolve(P_1, P_2)
print(P_1, P_2, Product, sep = '\n')


FP_1 = np.fft.fft(P_1)
FP_2 = np.fft.fft(P_2)
Product2 = np.array( [ FP_1[i] * FP_2[i] for i in range(0, len(FP_1)) ]  )

print( np.fft.ifft( Product2 )  )
