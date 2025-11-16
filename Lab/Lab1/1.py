import numpy as np
import matplotlib.pyplot as plt

# label-uri la axe, ca sa stim la ce ne uitam ( si dupa mai mult timp )

# (a)
E1 = np.arange(0, 0.03, 0.0005)

# (b)
# plt.subplots( <nr_coloane>, <nr_linii>, ... )
fig, axs = plt.subplots(3,1)

x = lambda x,a:np.cos(a * np.pi * x + np.pi/3)

#E = np.linspace(0,1,1000)
#E1 = [x for x in range(0, 0.03, 0.0005) ]
#E1 = np.linspace(0, 0.03, 1000)


axs[0].plot( E1, x(E1, 520), "r-" )
axs[0].set_ylabel("x(timp)")

axs[1].plot( E1, x(E1, 280), "g-" )
axs[1].set_ylabel("y(timp)")

axs[2].plot( E1, x(E1, 120), "b-" )
axs[2].set_ylabel("z(timp)")

for ax in axs.flat:
    ax.set_xlabel("timp")
    #ax.set_ylabel("fctia_semnal(timp)")

plt.show()
fig.savefig("fig_Exc1_(b)")


# (c)
fig, axs = plt.subplots(3, 1)

# E2 = np.linspace(0, 0.03, 200) -- Gresit; 200 Hz = 200 ori/secunda [ intervalul [0,1] ]
E2 = np.linspace(0, 0.03, 6)

#axs[0].plot( E2, x(E2, 520), "ro" )
#axs[1].plot( E2, x(E2, 280), "go" )
#axs[2].plot( E2, x(E2, 120), "bo" )

axs[0].stem( E2, x(E2, 520) )
axs[0].set_ylabel("x(n)")

axs[1].stem( E2, x(E2, 280) )
axs[1].set_ylabel("y(n)")

axs[2].stem( E2, x(E2, 120) )
axs[2].set_ylabel("z(n)")

for ax in axs.flat:
    ax.set_xlabel("n")
#    ax.set_ylabel("fctia_semnal(timp)")

plt.show()
fig.savefig("fig_Exc1_(c)")
