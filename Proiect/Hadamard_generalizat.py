import numpy as np
import math
import matplotlib.pyplot as plt
# Operatori pe biti!
# np.reshape()
# np.vstack( (<o_singura_chestie>,) * <de_cate_ori_sa_se_multiplice> )
"""
def add_noise(v, nr_erori):
    eps = [0 for i in range(0,len(v))]
    pozitii = np.random.randint(low = 0, high = len(v), size = nr_erori)
    for i in pozitii:
        eps[i] = 1
    return v + np.array(eps)
"""
def add_noise(v, nr_erori):
    eps = [0 for i in range(0,len(v))]
    rng = np.random.default_rng()
    pozitii = rng.choice(len(v), size = nr_erori, replace = False) 
    for i in pozitii:
        eps[i] = 1
    return v + np.array(eps)

# Vrem sa facem matricea generator pt. Codul Hadamard generalizat [2^r, r, 2^{r-1}]_2
    # G_r -- matrice 2^r x r
    
    # MERGE!!
def generator_matrix(r):
    numbers = [i for i in range(0,2**r)]
    bit_mask = 1
    #nr_of_symbols_needed = r
    
    G_r = []
    for nr in numbers:
        row = [0 for i in range(0,r)]
        # Punem masca, a.i. sa gasim reprezentarea in baza 2 a  nr
        for i in range(0,r):
            #print(f"bit_mask & nr = {bit_mask & nr}")
            #print(f"(bit_mask & nr) << i = { (bit_mask & nr) << i }")
            #print()
            if (bit_mask & nr) >> i == 1:
                row[-i-1] = 1
            bit_mask = bit_mask << 1
        G_r.append(row)
        bit_mask = 1

    return np.array(G_r)

def error_correct(r, nr_erori, printeaza = 1):

        # Matr. Generator G_3
    G_r = generator_matrix(r)


        # G_3 @ (matr. cu coloanele = toate elem. din F_2^r (spatiul mesajelor pre-codificare)
            # adica matr. cu coloanele = codeword-urile asociate ^ de mai sus

    #mesaj = np.array([1 for i in range(0,r)])
    mesaj = np.random.randint(low = 0, high = 2, size = r)
    mesaj_encoded = (G_r @ mesaj) % 2
    mesaj_encoded_zgomotos = add_noise(mesaj_encoded, nr_erori = nr_erori) % 2

    M_codewords = ( G_r @ G_r.transpose() ) % 2
        # MERGE! (matricea cu 2**r coloane de lungime 2**r, care-s toate mesaj_encoded_zgomotos)
    M_diferente = (  M_codewords -  np.hstack( (mesaj_encoded_zgomotos,) * 2**r).reshape((2**r,2**r)).transpose()  ) % 2

    M_distante = [sum(row) for row in M_diferente.transpose()]

    pozitia_minimului = M_distante.index(min(M_distante))
    if printeaza == 1:
        print(f"r = {r}\nnr_erori_zgomot = {nr_erori}")
        print(f"G_Ham =\n{G_r}")

        print(f"mesaj =\n{mesaj}")
        print(f"mesaj_encoded =\n{mesaj_encoded}")
        print(f"mesaj_encoded_zgomotos =\n{mesaj_encoded_zgomotos}")
        print(f"Matricea cu coloanele codeword-uri =\n{M_codewords}")

        print(f"Matricea cu diferente =\n{M_diferente}")
        print(f"Ponderile Hamming ale coloanelor =\n{M_distante}")
        print(f"min = {min(M_distante)}, pozitia_minimului = {pozitia_minimului}")
        print(f"deci, mesajul original era coloana pozitia_minimului din matricea Codeword-urilor:\n{M_codewords[:,pozitia_minimului]}")

        print(f"Coloana {pozitia_minimului+1} == mesaj_encoded:\n{M_codewords[:,pozitia_minimului] == mesaj_encoded}")
    #vector_erori_necorectate = (M_codewords[:,pozitia_minimului] == mesaj_encoded)
    vector_erori_necorectate = (M_codewords[:,pozitia_minimului] + mesaj_encoded) % 2

    return vector_erori_necorectate 


#error_correct(4,3)
error_correct(4,16)

def simulare_n_corectari(n, r, nr_erori, printeaza = 1):
    Sum = 0
    for i in range(0,n):
        vector_erori_necorectate = error_correct(r, nr_erori, printeaza = 0),
        nr_erori_necorectate = sum(vector_erori_necorectate)
        nr_erori_necorectate = sum(nr_erori_necorectate)
        #nr_erori_necorectate = np.unique(vector_erori_necorectate, return_counts=True)[1]
        #unique, counts =  np.unique(vector_erori_necorectate, return_counts=True)[1]
        #Dictionar = dict( zip(unique, counts) )

        #print(f"nr_erori_necorectate = {nr_erori_necorectate}") 
        #print(f"unique = {unique}") 
        if printeaza == 1:
            print(f"vector_erori_necorectate = {vector_erori_necorectate}")
            print(f"nr_erori_necorectate = {nr_erori_necorectate}") 
            print()
        Sum += nr_erori_necorectate
    avg = Sum / n
    return avg

r = 4
n1 = 5 
n2 = 10000
#n3 = 1000

E = np.array([i for i in range(0,2**r+1)])

fig, axs = plt.subplots(1,2)
#fig.suptitle("Numarul erorilor ramase necorectate dupa error_correct\nin functie de numarul erorilor introduse de zgomot\n(in medie, dupa N simulari)") 

axs[0].set_xlabel("nr. erori introduse de zgomot")
axs[0].set_ylabel("nr. erori necorectate (medie dupa N simulari)")
axs[0].set_title(f"N = {n1}, r = {r}")
axs[0].set_xticks([i for i in range(0,2**r+1)])

axs[1].set_xlabel("nr. erori introduse de zgomot")
axs[1].set_ylabel("nr. erori necorectate (medie dupa N simulari)")
axs[1].set_title(f"N = {n2}, r = {r}")
axs[1].set_xticks([i for i in range(0,2**r+1)])

"""
axs[2].set_xlabel("nr. erori introduse de zgomot")
axs[2].set_ylabel("nr. erori necorectate (medie dupa N simulari)")
axs[2].set_title(f"N = {n3}, r = {r}")
axs[2].set_xticks([i for i in range(0,2**r+1)])
"""

"""
axs[0].plot(E, np.array([simulare_n_corectari(n1, r, item, 0) for item in E]), "r-")
axs[1].plot(E, np.array([simulare_n_corectari(n2, r, item, 0) for item in E]), "r-")
axs[2].plot(E, np.array([simulare_n_corectari(n3, r, item, 0) for item in E]), "r-")
"""
axs[0].stem(E, np.array([simulare_n_corectari(n1, r, item, 0) for item in E]))
axs[1].stem(E, np.array([simulare_n_corectari(n2, r, item, 0) for item in E]))
#axs[2].stem(E, np.array([simulare_n_corectari(n3, r, item, 0) for item in E]))

plt.savefig("plot_erori_ramase_necorectate.pdf", format = "pdf")
plt.show()


