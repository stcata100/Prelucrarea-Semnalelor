import numpy as np
import matplotlib.pyplot as plt
import math




c_exp = lambda w, k, N: math.e**(-2 * math.pi * 1j * (w/N) * k)


        # Exc. 1:
    # Matricea Fourier
N = 8
#M = np.array( [ [c_exp(col, lin, N) for lin in range(0,N)] for col in range(0,N)  ] )
fourier_matrix = lambda dim : np.array( [ [c_exp(col, lin, dim) for lin in range(0,dim)] for col in range(0,dim)  ] )


M = fourier_matrix(N)

    # Verificam daca este unitara
hermitian = lambda M : np.transpose( np.conj(M) )

P = np.matmul(M, hermitian(M))
print( np.allclose( P[0][0]*np.eye(N), P )  )



M_H = np.transpose( np.conj( M ) )
        # --------



