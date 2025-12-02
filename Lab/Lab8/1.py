import matplotlib.pyplot as plt
import numpy as np

# np.random.normal( left, right, cate_valori ) ---- 
# print( np.random.normal(0, 1, 10) )

# (a)
E = np.array( [i for i in range(0, 1000)] )
E1 = np.linspace(1, 7, 1000)
E2 = np.linspace(0, 1, 1000)
f = lambda x : x**2
x = lambda A, freq, t, shift : A * np.sin( 2*np.pi*freq*t + shift )

Trend = f(E1)
Sine1 = x(2, 1/(1/4), E2, 0)
Sine2 = x(8, 1/(1/3), E2, 0)
Sezon = np.random.normal( -1, 1, 1000 )

SerieDeTimp = Trend + Sine1 + Sine2 + Sezon

fig, axs = plt.subplots(5, 1)
axs[0].plot(E, Trend, "r-")
axs[1].plot(E, Sine1, "g-")
axs[2].plot(E, Sine2, "b-")
axs[3].plot(E, Sezon, "y-")
#axs[4].plot(E, Trend + Sine1 + Sine2 + Sezon, "c-")
# e plotata mai jos a.i. plot-ul ei sa fie deasupra plot-ului variantei ei cu predictii date de modelul AR

for ax in axs.flat:
    ax.set_xlabel("timp")
axs[0].set_ylabel("fctie de grad 2")
axs[1].set_ylabel("sinusoidala1")
axs[2].set_ylabel("sinusoidala2")
axs[3].set_ylabel("zgomot")
axs[4].set_ylabel("serie de timp (suma lor)")

#plt.show()

# (b)

#   s.n. vector de autocorelatie pt. ca pe fiecare componenta a sa avem
# corelatia(produsul scalar al) vectorului y cu (y shiftat cu indicele_randului pozitii) ??
N = len(SerieDeTimp)
m = 900 # Obs: nu putem alege m > (lungimea SerieDeTimp)
p = N - m # p ales a.i. sa fie maxim

y = SerieDeTimp[-1:-1-m:-1] # ultimele m elemente din SerieDeTimp

Y = np.array(  [ SerieDeTimp[-1-k : -1-m-k : -1] for k in range(0, p) ]  ).transpose()
# coloanele lui Y = alegerea de m elemente, shiftata cu k in {0, ..., p}

#VectDeCorelatie = np.matmul( Y.transpose(), y )
VectDeCorelatie = Y.transpose() @ y

# (c)
# Obs: matrice * matrice ---- elementwise multiplication <=> np.multiply()
#       matrice @ matrice ---- matrix multiplication <=> np.matmul()

# Rezolvam problema de least squares
x = np.linalg.inv(Y.transpose() @ Y) @ VectDeCorelatie

catiTermeniSaAproximam = 10
for i in range(0, catiTermeniSaAproximam):
    SerieDeTimp = np.append( SerieDeTimp, x @ SerieDeTimp[-1 : -1-p : -1] ) # appendam o aproximare in fctie de ultimii p termeni ai SerieDeTimp; facem asta de catiTermeniSaAproximam ori
    print( x @ SerieDeTimp[-1 : -1-p : -1] )

# SerieDeTimp = np.append( SerieDeTimp, [i for i in range(1, 101)] )

axs[4].plot( np.array([i for i in range(0, len(SerieDeTimp))]), SerieDeTimp, "r-" )
axs[4].plot(E, Trend + Sine1 + Sine2 + Sezon, "c-")
plt.show()

