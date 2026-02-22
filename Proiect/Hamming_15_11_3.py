import numpy as np

def encode(mesaj):
    # Generator matrix(GRESIT! e de fapt Parity-Check-Matrix)
    #G = np.array([[0,0,0,1],[0,1,0,0],[0,0,1,1],[0,1,0,0],[0,1,0,1],[0,1,1,0],[0,1,1,1],[1,0,0,0],[1,0,0,1],[1,0,1,0],[1,0,1,1],[1,1,0,0],[1,1,0,1],[1,1,1,0],[1,1,1,1]])

        # Parity check matrix pt. [15,11,3]_2 Hamming code
    H_systematic = np.array([[1,1,1,1],[1,1,1,0],[0,0,1,1],[1,1,0,1],[0,1,0,1],[0,1,1,0],[0,1,1,1],[1,1,0,0],[1,0,0,1],[1,0,1,0],[1,0,1,1],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]).transpose()

        # "matricea -P (= P pt ca este peste F_2)" 
    P = H_systematic[:,:11]

        # Matricea Generator  obtinuta ca si (I_k | P)^transpus
    G = np.vstack(  (np.eye( P.shape[1] ), P)  )

    return (G @ mesaj) % 2
    #print(G)
    #print( (H_systematic @ G) % 2 )
    
#    print(H)
#    print(H[H.ndim * [slice(0,11)] ])
#    print(H[slice(1,-1)])
#    print(H[:,:11])
        #Obs.: G @ T^G ) % 2 = O (in F_2 matr. G^transpus este Parity check matrix pt. matricea G) GRESIT!!!! 

    
    #print( (G.transpose() @ G) % 2 )
def flip_one_bit(vector):
    #return (  vector + np.array( [1 if i = np.random.randint(0, len(vector), 1) else 0 for i in range(0,len(vector))] )  ) % 2
    eps = [0 for i in range(0,len(vector))]
    eps[ np.random.randint(0,len(vector)) ] = 1
    return (vector + np.array(eps) ) % 2

def error_correction(v):
    H_systematic = np.array([[1,1,1,1],[1,1,1,0],[0,0,1,1],[1,1,0,1],[0,1,0,1],[0,1,1,0],[0,1,1,1],[1,1,0,0],[1,0,0,1],[1,0,1,0],[1,0,1,1],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]).transpose()

    syndrome = (H_systematic @ v) % 2
    print(f"syndrome = {syndrome}")
#    index_of_column = np.where( H_systematic = syndrome ) 
    H_transpose = H_systematic.transpose()
    for i in range(0, len(H_transpose)):
        if (H_transpose[i] == syndrome).all(): # H^T == syndrome  --> [true, ... true]; [true, ..., true].all() --> true.
            print(f"pozitia unde a avut loc eroarea = {i + 1}")
            v[i] = (v[i] + 1) % 2
    #print(f"index_of_column = {index_of_column}")
    return v

def decode(c):
    return c[0:11]


#m = [1 for i in range(0,11)]
m = np.random.randint(0, 2, 11)
print(f"mesaj = \n{m}")
m_encoded = encode(m)
m_encoded_and_flipped = flip_one_bit(m_encoded)
print(f"mesaj codificat = \n{m_encoded}")
print(f"mesaj codificat si flipped = \n{m_encoded_and_flipped}")
m_corrected = error_correction(m_encoded_and_flipped)
print(f"mesaj codificat corectat = \n{m_corrected}")
print(f"mesaj corectat decodificat = \n{decode(m_corrected)}")





