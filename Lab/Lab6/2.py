import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# (2)

x = np.random.rand(100)

"""

#conv = lambda n, X, H : sum( [ H[i] * X[n-i] for i in range(0, len(H)) ] )

#conv_returns_vector = lambda X, H :  [ sum( [ H[i] * X[n-i] for i in range(0,len(H)) ] ) for n in range(0, len(X)) ]

#conv_returns_vector = lambda X, H :  [ sum( [ H[i] * X[n-i] for i in range(0,len(H)) ] ) for n in range(0, len(X)) ]
"""

# Convolutie "discret"

def conv_returns_vector(X, H):
    Y = np.zeros( len(X) )
    
    for k in range(0, len(H)):
        for n in range(0, len(X)):
            if n >= k:
                Y[n] += H[k] * X[n-k]


    return Y

def convolve3(X, H):
    Y = np.zeros( len(X) + len(H) - 1 )

    for k in range(0, len(H)):
        for n in range(0, len(X)):
            # Obs: converge la Gaussiana si fara "if"
            # Q: "e nevoie sau nu de "if"?"
            if n >= k:
                Y[n] += H[k] * X[n-k]

    return Y
"""
#fig, axs = plt.subplots(3, 1)

# convolutie cu sine insusi si plotare
# E = np.linspace(0, 1, len(x))

for i in range(0,3):
    X = x
    for j in range(0, i+1):
        X = np.array( [conv(elem, X, x) for elem in E] )

    axs.flat[i].plot( E, X, "r-" )
"""

def plot_self_convolve(x, times = 3):
    
    fig, axs = plt.subplots(times+1, 1)
    # Plotam semnalul original
    axs[0].plot( np.linspace(0, 1, len(x)), x / np.linalg.norm(x), "b-")
    axs[0].set_xlabel("timp")
    axs[0].set_ylabel("semnalul original")

    for i in range(0, times):
        X = x
        for j in range(0,i+1):
            #X = conv_returns_vector(X, x)
            #X = np.convolve(X, x)
            X = convolve3(X, x)

        axs.flat[i+1].plot( np.linspace(0, 1, len(X)) , X/np.linalg.norm(X), "r-" )
        axs.flat[i+1].set_xlabel("timp")
        #axs.flat[i+1].set_ylabel(" $\\ast\\{n=1}^{i} (semnalul original)$ ")
        axs.flat[i+1].set_ylabel(" $ \\underset{n=1}{ \\overset{i}{ \\ast } } (semnalul original) $")

    #axs[0].plot( np.linspace(0, 1, len(x)), x/np.linalg.norm(x), "b-")
    plt.show()

"""
################ teste:
#fig, axs = plt.subplots(3,1)

E = np.linspace(0, 1, 100)
#print(square)

#axs[0].plot( E, square )
#axs[1].plot( E, conv_returns_vector(square, np.array([1/2, -1/2]) )  )


#axs[1].plot( E_c, C/np.linalg.norm(C), "r-")
#axs[2].plot( E, conv_returns_vector(square, square) )
#axs[2].plot( E, conv_returns_vector(square, np.array([0,0,0,1])), "r-")
#axs[2].plot( E, square, "b-")

#axs[2].plot( E, np.convolve(square, square) )
#   Da eroare: "size mismatch pt. ca np.convolute() returneaza un vector de lungime 
# diferita de len(X)

#plt.show()
"""
square =  [ (i // 20) % 2 for i in range(0,100) ]

plot_self_convolve(square, 8)
plot_self_convolve(x, 8)




        

