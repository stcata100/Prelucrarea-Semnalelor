import numpy as np
import matplotlib.pyplot as plt

# Q: care-i seria Fourier a graficului patrat

# (a)

x = lambda t,freq,shift : np.sin( freq * 2 * np.pi * t + shift )

fig, axs = plt.subplots(4, 1)

for ax in axs.flat:
    ax.set_xlabel("timp")
#    ax.set_ylabel("fctia_semnal(timp)")

E = np.linspace(0, 0.03, 1600)
#axs[0].plot( E, x(E, 400*0.03, 0) , "r-" )   gresit pt. ca avea frecventa = 400*0.3, nu 400
axs[0].plot( E, x(E, 400, 0) , "r-" )
axs[0].set_ylabel("f=400Hz, f_s=1600Hz")

# (b)
T = np.linspace(0,3,20000)
Im_T = x(T, 800, 0)

axs[1].plot( T[0:200], Im_T[0:200] )
axs[1].set_ylabel("f=800Hz, durata=3s")

print(Im_T[0] - Im_T[-1])
# Q: ^ => se "ia si ultimul punct = primul pct" ??
# (c)
f = lambda x,freq:freq*x
axs[2].plot( E, f(E,240) - np.floor( f(E,240) )  )
axs[2].set_ylabel("sawtooth, f=240Hz")


# completare:
# (d)
f2 = lambda x,ampl,freq,shift : np.sign(  ampl * np.sin( 2*np.pi*x * freq + shift )  )
axs[3].plot( E, f2(E,0.5,300,0) + 1 )
axs[3].set_ylabel("square, f=300Hz")

# (e)
a = np.random.rand( 128,128 )
    # sampling al [-3,3]x[-3,3] cu 128x128 esantioane egal departate:
a2 = np.array( [ np.linspace(-3,3,128) for i in range(0,128) ] )

#a2 = np.array( [ (i,j) for i in np.linspace(-1, 1, 128) for j in np.linspace(-1, 1, 128) ] )
#a2.reshape( (128,128) )
#print(len( np.linspace(-1,1,128) ) )  --> 128
#print( a2.shape() )


#print(a2[0,0], a2[1,1] )
#print( np.linspace(-1, 1, 10) )
#print( np.array( [ (i,j) for i in np.linspace(-1, 1, 10) for j in np.linspace(-1,1,10) ] )  )

fig2, ax = plt.subplots(1,4)

ax[0].imshow(a)
ax[0].set_xlabel("semnal 2D aleator")

ax[1].imshow(a2)
ax[1].set_xlabel("semnal 2D $f(x,y) := x, \\forall (x,y) \\in [-3,3]\\times[-3,3]$")


# (f)
ax[2].imshow(a2**2)
ax[2].set_xlabel("semnal 2D $f(x,y) := x^2 , \\forall (x,y) \\in [-3,3]\\times[-3,3]$")

ax[3].imshow( np.array( [ [ (i+j)%2 for i in range(0,128) ] for j in range(0,128) ] )    )
ax[3].set_xlabel("semnal 2D \"chessboard\"")


plt.show()
fig.savefig("fig1_Exc2_(a)-(d)")
fig2.savefig("fig2_Exc2_(e)-(f)")



