import numpy as np
import matplotlib.pyplot as plt

# Q: care-i seria Fourier a graficului patrat

# (a)

x = lambda t,freq,shift : np.sin( freq * 2 * np.pi * t + shift )

fig, axs = plt.subplots(4, 1)

for ax in axs.flat:
    ax.set_xlabel("timp")
    ax.set_ylabel("fctia_semnal(timp)")

E = np.linspace(0, 0.03, 1600)
axs[0].plot( E, x(E, 400*0.03, 0) , "r-" )

# (b)
T = np.linspace(0,3,20000)
Im_T = x(T, 800, 0)

axs[1].plot( T[0:200], Im_T[0:200] )

# (c)
f = lambda x,freq:freq*x
axs[2].plot( E, f(E,240) - np.floor( f(E,240) )  )

# (d)
f2 = lambda x,ampl,freq,shift : np.sign(  ampl * np.sin( 2*np.pi*x * freq + shift )  )
axs[3].plot( E, f2(E,0.5,300,0) + 1 )

# (e)
#a = np.array

plt.show()



