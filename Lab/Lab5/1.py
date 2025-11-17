import matplotlib.pyplot as plt
import numpy as np
import csv




X = np.genfromtxt("Train.csv", delimiter=",")
#print(X[:14])
print(f"nr_esantioane = len(X) = {len(X)}")

# (a)
print(f"f_s =?= 1/3600 cycles/s")

# (b)
print(f" t_final = nr_esantioane * 3600 s = {len(X) * 3600} s")

fig, (ax_timp,ax_frecv) = plt.subplots(1,2)

ax_timp.set_xlabel("timp")
ax_timp.set_ylabel("esantioane")

ax_frecv.set_xlabel("frecventa")
ax_frecv.set_ylabel("corelatie")

ax_timp.plot( [ x[0] for x in X ], [ x[2] for x in X], "b-" )

# (d)
FX = np.fft.fft( np.array([ x[2] for x in X[1:] ]) )


f_s = 1/3600
N = len(FX)
ax_frecv.plot( f_s * np.linspace(0, N//2, N//2) / N, [np.abs(z) for z in FX[:N//2]  ], "r-" )

plt.show()


