import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# (1)

x = lambda t, B : np.sinc( B * t )**2
x_hat = lambda t, X, T_s : sum( [ X[i] * np.sinc( (t - i*T_s)/T_s ) for i in range(0,len(X)) ]   )

# sa afisez sinc-ul ce corespunde pt. i = 0



E_t = np.linspace(-3, 3, 1600)

fig, axs = plt.subplots(4,1)

for ax in axs.flat:
    ax.plot( E_t, x(E_t, 1), "g-" )
    ax.set_xlabel("t[s]")
    ax.set_ylabel("Amplitudine")
# f_s = 1
E_1 = np.concatenate( [ np.linspace(-3, 0, 4), np.linspace(0, 3, 4)[1:] ] )
# Obs: trb 4 esantioane pe [-3,0] sau [0,3], ca sa avem "3 perioade de lungime 1" 
# (gardurile si parii)
# In general, trb len(Interval)/n + 1 esantioane pt a a avea "n perioade de lungime 1"

# f_s = 1,5 
#E_2 = np.concatenate( [ np.linspace(-3, 0, 9/2+1), np.linspace(0, 3, 9/2+1)[1:] ] )

# f_s = 2 
E_3 = np.concatenate( [ np.linspace(-3, 0, 6+1), np.linspace(0, 3, 6+1)[1:] ] )
# f_s = 4
E_4 = np.concatenate( [ np.linspace(-3, 0, 12+1), np.linspace(0, 3, 12+1)[1:] ] )


axs[0].stem( E_1, x(E_1, 1), "o-" )
#axs[1].stem( E_2, x(E_2, 1) )
axs[2].stem( E_3, x(E_3, 1) )
axs[3].stem( E_4, x(E_4, 1) )

axs[0].plot( E_t, x_hat(E_t, x(E_1, 1), 1), "r--" )
#axs[1].plot( E_t, x_hat(E_t, x(E_2, 1), 1/1.5), "r-" )
axs[2].plot( E_t, x_hat(E_t, x(E_3, 1), 1/2), "r--" )
axs[3].plot( E_t, x_hat(E_t, x(E_4, 1), 1/4), "r--" )




plt.show()



