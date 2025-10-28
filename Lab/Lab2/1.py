import matplotlib.pyplot as plt
import numpy as np


# (1)
x1 = lambda A, f, t, shift : A*np.sin( 2*np.pi * f * t  + shift )
x2 = lambda A, f, t, shift : A*np.cos( 2*np.pi * f * t  + shift )

fig, axs = plt.subplots(1, 2)

E = np.linspace(0, 1, 1600)

axs[0].plot( E, x1( 3, 1, E, 0 ), "g-" )
axs[0].set_xlabel("timp")
axs[0].set_ylabel("sinusoidala sin")
axs[1].plot( E, x2( 3, 1, E, -np.pi/2 ), "r-" )
axs[1].set_xlabel("timp")
axs[1].set_ylabel("sinusoidala cos")

plt.show()

# (2)

fig, axs = plt.subplots(1,1)

axs.plot( E, x1( 1, 3, E, 0 ), "r-" )
axs.plot( E, x1( 1, 3, E, np.pi/4 ), "g-" )
axs.plot( E, x1( 1, 3, E, np.pi/2 ), "b-" )
axs.plot( E, x1( 1, 1, E, (3*np.pi)/4 ), "y-" )

#for shift in [0, np.pi/4, np.pi/2, (3*np.pi)/2 ]:
#    axs.plot( E, x1( 1, 3, E, shift )

axs.set_xlabel("timp")
axs.set_ylabel("semnale cu faze diferite")

plt.show()

# print( np.random.normal(0, 1, 10) )

X = x1( 1, 3, E, 0 )
Z = np.random.normal(0,1, len(X) )

gamma = [ np.sqrt( (np.linalg.norm(X)**2 / np.linalg.norm(Z)**2) / k ) for k in [0.1, 1, 10, 100] ]

print( np.linalg.norm(X), np.linalg.norm(Z), np.linalg.norm(X)/np.linalg.norm(Z) )

fig, axs = plt.subplots(1,1)
axs.plot( E, X + gamma[0] * Z, "b-" )
axs.plot( E, X + gamma[1] * Z, "g-" )
axs.plot( E, X + gamma[2] * Z, "y-" )
axs.plot( E, X + gamma[3] * Z, "m-" )
axs.plot( E, X, "r-" )

axs.set_xlabel("timp")
axs.set_ylabel("semnal + noise-cu SNR $\\in {0.1, 1, 10, 100} $ " )

plt.show()

