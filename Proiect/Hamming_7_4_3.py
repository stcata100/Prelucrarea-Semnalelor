import numpy as np

# [7,4,3]_2 Hamming code

    # Generator matrix
G = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1], [0,1,1,1], [1,0,1,1], [1,1,0,1]])
    # Parity check matrix al ^
H = np.array( [[0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]] ).transpose()

    # Generam (vector de) zgomot 
eps = np.random.randint(low = 0, high = 2, size = 7) 

def generate_noise_vector(size = 7, nr_of_errors = 1):
    v_pozitii = np.random.randint(0, size, nr_of_errors)
    return np.array( [1 if i in v_pozitii else 0 for i in range(0,size)] )


    # Mesajul
v = [1,0,1,0]
    #Encodarea Mesajului v
v1 = (G@v) % 2
print(f"mesaj = {v}")
print(f"mesaj_encoded = {v1}")
    #"(Simulam) Transmiterea mesajului v1 printr-un canal zgomotos (adaugam zgomot de Hamming_weight ales[ales =1 pt ca altfel "min.distance property => nu putem corecta eroarea"])"
w1 = ( v1 + generate_noise_vector(len(v1), 1) ) % 2
print(f"mesaj_encoded_zgomotos = {w1}")
    # Gasim H@eps (pt. ca H@w1 = H@(v1+eps) = 0 + H@eps)
Heps = (H @ w1) % 2
print(f"syndrome = {Heps}")
pozitia_erorii = sum( [2**(2-i) if Heps[i] == 1 else 0 for i in range(0,3)] )
print(f"deci, eroare pe pozitia: {pozitia_erorii}")

w1[pozitia_erorii - 1] = (w1[pozitia_erorii - 1] + 1) % 2
print(f"mesaj_encoded_corectat = {w1}")

print(f"mesaj_corectat_decoded (primele 4 componente, pt. ca G e in \"systematic form\":\n {[w1[i] for i in range(0,4)]}")







