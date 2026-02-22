import matplotlib.pyplot as plt
import numpy as np

X = np.linspace(0,100,1000)
rate_Hamming = lambda r : (2**(r) - 1 - r) / 2**r
rate_Hadamard = lambda r : r / (2**r)

mindist_Hamming = lambda x : 3
mindist_Hadamard = lambda r : 2**(r-1)

fig, axs = plt.subplots(1,2)
fig.suptitle("Hamming - rosu, Hadamard - albastru")

axs[0].set_title("Rata")
axs[0].set_xlabel("r")
axs[0].set_ylabel("Rata")

axs[1].set_title("Distanta minima")
axs[1].set_xlabel("r")
axs[1].set_ylabel("Distanta minima")

axs[0].plot(X, rate_Hamming(X), "r-")
axs[0].plot(X, rate_Hadamard(X), "b-")

axs[1].plot(X, [3 for i in range(0,len(X))], "r-")
axs[1].plot(X, mindist_Hadamard(X), "b-")


plt.savefig("plot_rata_si_mindistance.pdf", format = "pdf")
plt.show()







