import matplotlib.pyplot as plt
import numpy as np

# ramase: cele cu sounddevice (3) si (5); observatii la (5), (6), (7)

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

#plt.show()
fig.savefig("fig_Exc1")
# (2)

fig2, axs = plt.subplots(1,1)

axs.plot( E, x1( 1, 3, E, 0 ), "r-" )
axs.plot( E, x1( 1, 3, E, np.pi/4 ), "g-" )
axs.plot( E, x1( 1, 3, E, np.pi/2 ), "b-" )
axs.plot( E, x1( 1, 1, E, (3*np.pi)/4 ), "y-" )

#for shift in [0, np.pi/4, np.pi/2, (3*np.pi)/2 ]:
#    axs.plot( E, x1( 1, 3, E, shift )

axs.set_xlabel("timp")
axs.set_ylabel("semnale cu faze diferite")

#plt.show()
fig2.savefig("fig1_Exc2")

# print( np.random.normal(0, 1, 10) )

X = x1( 1, 3, E, 0 )
Z = np.random.normal(0,1, len(X) )

gamma = [ np.sqrt( (np.linalg.norm(X)**2 / np.linalg.norm(Z)**2) / k ) for k in [0.1, 1, 10, 100] ]

#print( np.linalg.norm(X), np.linalg.norm(Z), np.linalg.norm(X)/np.linalg.norm(Z) )

fig3, axs = plt.subplots(1,1)
axs.plot( E, X + gamma[0] * Z, "b-" )
axs.plot( E, X + gamma[1] * Z, "g-" )
axs.plot( E, X + gamma[2] * Z, "y-" )
axs.plot( E, X + gamma[3] * Z, "m-" )
axs.plot( E, X, "r-" )

axs.set_xlabel("timp")
axs.set_ylabel("semnal + noise-cu SNR $\\in \{0.1, 1, 10, 100\} $ " )

#plt.show()
fig3.savefig("fig2_Exc2")


# (3)




# (4)
sawtooth = lambda t, freq : freq*t - np.floor(freq*t)

fig4, axs4 = plt.subplots(3,1)
axs4[0].plot( E, x1(1,2,E,0), "r-" )
axs4[1].plot( E, sawtooth(E, 10), "b-" )
axs4[2].plot( E, x1(1,2,E,0) + sawtooth(E, 10), "r-" )

for ax in axs4.flat:
    ax.set_xlabel("timp")
axs4[0].set_ylabel("semnal sinusoidal")
axs4[1].set_ylabel("semnal sawtooth")
axs4[2].set_ylabel("sinsuoidal + sawtooth")
#plt.show()
fig4.savefig("fig_Exc4")

# (5)



# (6)
fig5, axs5 = plt.subplots(3,1)
E5 = np.linspace(0, 1/200, 1600) # f_s = 1600 cycles/s
# am ales capatul din stanga al intervalului de timp = 1/200 ca sa 
# apara in plot 8 perioade ale unui semnal cu f=1600
# (a)
axs5[0].plot( E5, x1(1, 1600/2, E5, 0), "r-" )
# (b)
axs5[1].plot( E5, x1(1, 1600/4, E5, 0), "b-" )
# (c)
axs5[2].plot( E5, x1(1, 0, E5, 0), "g-" )
axs5[2].plot( E5, x1(1, 1600/4, E5, 0), "b-" )
axs5[2].plot( E5, x1(1, 1600/2, E5, 0), "r-" )
axs5[2].plot( E5, x1(1, 1600/4, E5, 0), "b-" )



for ax in axs5.flat:
    ax.set_xlabel("timp")
axs5[0].set_xlabel("sinusoida cu $A=1$, $f=\\frac{f_s}{2}$")
axs5[1].set_xlabel("sinusoida cu $A=1$, $f=\\frac{f_s}{4}$")
axs5[2].set_xlabel("sinusoida cu $A=1$, $f=0 Hz$")
#plt.show()
fig5.savefig("fig_Exc5")

# (7)
E7 = np.linspace(0, 8/240, 1000) 
    #8/240 ales a.i. sa se vada 8 perioade ale unui semnal cu f=240Hz

fig, axs = plt.subplots(3,1)
# (a)
axs[0].plot( E7, x1(1, 240, E7, 0), "r-" )
axs[1].plot( E7[::4], x1(1, 240, E7, 0)[::4], "b-" ) 
axs[2].plot( E7, x1(1, 240, E7, 0), "r-" )
axs[2].plot( E7[::4][1::4], x1(1, 240, E7, 0)[::4][1::4], "g-" ) 
# Obs: List[::4] pastreaza si primul element din lista

for ax in axs.flat:
    ax.set_xlabel("timp")
axs[0].set_ylabel("sinusoidala cu $f=240Hz$, $f_s=1000$")
axs[1].set_ylabel("sinusoidala cu $f=240Hz$, $f_s=\\frac{1000}{4}$")
axs[2].set_ylabel("sinusoidala cu $f=240Hz$, $f_s=\\frac{1000}{4^2}$")

#plt.show()
fig.savefig("fig_Exc7")




# (8)
E8_1 = np.linspace(-1/1000, 1/1000, 1600)
E8_2 = np.linspace(-1/100, 1/100, 1600)
E8_3 = np.linspace(-1/2, 1/2, 1600)

# Aprox. Taylor
fig, axs = plt.subplots(2,3)
axs[0][0].plot( E8_1, np.sin(E8_1), "r-" )
axs[0][0].plot( E8_1, E8_1, "b-" )
axs[1][0].plot( E8_1, np.sin(E8_1) - E8_1, "g-" )

axs[0][1].plot( E8_2, np.sin(E8_2), "r-" )
axs[0][1].plot( E8_2, E8_2, "b-" )
axs[1][1].plot( E8_2, np.sin(E8_2) - E8_2, "g-" )

axs[0][2].plot( E8_3, np.sin(E8_3), "r-" )
axs[0][2].plot( E8_3, E8_3, "b-" )
axs[1][2].plot( E8_3, np.sin(E8_3) - E8_3, "g-" )

for ax in axs.flat:
    ax.set_xlabel("timp")

axs[0][0].set_ylabel("$y=sin(t)$ red, $y=t$ blue, domeniul=$[-\\frac{1}{1000},\\frac{1}{1000}]$")
axs[0][1].set_ylabel("$y=sin(t)$ red, $y=t$ blue, domeniul=$[-\\frac{1}{100},\\frac{1}{100}]$")
axs[0][2].set_ylabel("$y=sin(t)$ red, $y=t$ blue, domeniul=$[-\\frac{1}{2},\\frac{1}{2}]$")

for ax in axs.flat[3:]:
    ax.set_ylabel("fctia $t\\to sin(t)-t$ pe domeniul corespunzator")

#plt.show()
fig.savefig("fig_Taylor_lienar_Exc8")

# Aprox. Taylor, scara log_10()
fig, axs = plt.subplots(2,3)
axs[0][0].plot( E8_1, np.sin(E8_1), "r-" )
axs[0][0].plot( E8_1, E8_1, "b-" )
axs[1][0].plot( E8_1, np.sin(E8_1) - E8_1, "g-" )

axs[0][1].plot( E8_2, np.sin(E8_2), "r-" )
axs[0][1].plot( E8_2, E8_2, "b-" )
axs[1][1].plot( E8_2, np.sin(E8_2) - E8_2, "g-" )

axs[0][2].plot( E8_3, np.sin(E8_3), "r-" )
axs[0][2].plot( E8_3, E8_3, "b-" )
axs[1][2].plot( E8_3, np.sin(E8_3) - E8_3, "g-" )

for ax in axs.flat:
    ax.set_xlabel("timp")

axs[0][0].set_ylabel("$y=sin(t)$ red, $y=t$ blue, domeniul=$[-\\frac{1}{1000},\\frac{1}{1000}]$")
axs[0][1].set_ylabel("$y=sin(t)$ red, $y=t$ blue, domeniul=$[-\\frac{1}{100},\\frac{1}{100}]$")
axs[0][2].set_ylabel("$y=sin(t)$ red, $y=t$ blue, domeniul=$[-\\frac{1}{2},\\frac{1}{2}]$")

for ax in axs.flat[3:]:
    ax.set_ylabel("fctia $t\\to sin(t)-\\frac{t-\\frac{7t^3}{60}}{1+\\frac{t^2}{20}}$ pe domeniul corespunzator, scara $log_{10}$")
    ax.set_yscale("log")
fig.savefig("fig_Taylor_log_Ecx8")

# Aprox Pade
Pade = lambda t : (t - (7 * t**3 / 60)) / (1 + (t**2/20))

fig, axs = plt.subplots(2,3)
axs[0][0].plot( E8_1, np.sin(E8_1), "r-" )
axs[0][0].plot( E8_1, Pade(E8_1), "b-" )
axs[1][0].plot( E8_1, np.sin(E8_1) - Pade(E8_1), "g-" )

axs[0][1].plot( E8_2, np.sin(E8_2), "r-" )
axs[0][1].plot( E8_2, Pade(E8_2), "b-" )
axs[1][1].plot( E8_2, np.sin(E8_2) - Pade(E8_2), "g-" )

axs[0][2].plot( E8_3, np.sin(E8_3), "r-" )
axs[0][2].plot( E8_3, Pade(E8_3), "b-" )
axs[1][2].plot( E8_3, np.sin(E8_3) - Pade(E8_3), "g-" )

for ax in axs.flat:
    ax.set_xlabel("timp")

axs[0][0].set_ylabel("$y=sin(t)$ red, $y=\\frac{t-\\frac{7t^3}{60}}{1+\\frac{t^2}{20}}$ blue, domeniul=$[-\\frac{1}{1000},\\frac{1}{1000}]$")
axs[0][1].set_ylabel("$y=sin(t)$ red, $y=\\frac{t-\\frac{7t^3}{60}}{1+\\frac{t^2}{20}}$ blue, domeniul=$[-\\frac{1}{100},\\frac{1}{100}]$")
axs[0][2].set_ylabel("$y=sin(t)$ red, $y=\\frac{t-\\frac{7t^3}{60}}{1+\\frac{t^2}{20}}$ blue, domeniul=$[-\\frac{1}{2},\\frac{1}{2}]$")

for ax in axs.flat[3:]:
    ax.set_ylabel("fctia $t\\to sin(t)-\\frac{t-\\frac{7t^3}{60}}{1+\\frac{t^2}{20}}$ pe domeniul corespunzator")
fig.savefig("fig_Pade_Exc8")
 
# Aprox. Pade, scara log
fig, axs = plt.subplots(2,3)
axs[0][0].plot( E8_1, np.sin(E8_1), "r-" )
axs[0][0].plot( E8_1, Pade(E8_1), "b-" )
axs[1][0].plot( E8_1, np.sin(E8_1) - Pade(E8_1), "g-" )

axs[0][1].plot( E8_2, np.sin(E8_2), "r-" )
axs[0][1].plot( E8_2, Pade(E8_2), "b-" )
axs[1][1].plot( E8_2, np.sin(E8_2) - Pade(E8_2), "g-" )

axs[0][2].plot( E8_3, np.sin(E8_3), "r-" )
axs[0][2].plot( E8_3, Pade(E8_3), "b-" )
axs[1][2].plot( E8_3, np.sin(E8_3) - Pade(E8_3), "g-" )

for ax in axs.flat:
    ax.set_xlabel("timp")

axs[0][0].set_ylabel("$y=sin(t)$ red, $y=\\frac{t-\\frac{7t^3}{60}}{1+\\frac{t^2}{20}}$ blue, domeniul=$[-\\frac{1}{1000},\\frac{1}{1000}]$")
axs[0][1].set_ylabel("$y=sin(t)$ red, $y=\\frac{t-\\frac{7t^3}{60}}{1+\\frac{t^2}{20}}$ blue, domeniul=$[-\\frac{1}{100},\\frac{1}{100}]$")
axs[0][2].set_ylabel("$y=sin(t)$ red, $y=\\frac{t-\\frac{7t^3}{60}}{1+\\frac{t^2}{20}}$ blue, domeniul=$[-\\frac{1}{2},\\frac{1}{2}]$")

for ax in axs.flat[3:]:
    ax.set_ylabel("fctia $t\\to sin(t)-\\frac{t-\\frac{7t^3}{60}}{1+\\frac{t^2}{20}}$ pe domeniul corespunzator, scara $log_{10}$")
    ax.set_yscale("log")
fig.savefig("fig_Pade_log_Exc8")

plt.show()
