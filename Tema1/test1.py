import numpy as np
import matplotlib.pyplot as plt
    # Q: "ce inseamna astea de mai jos"??
        # dctn := "discrete cosine transform -- n dimensional" (?)
        # idctn := "INVERSE discrete cosine transform -- n dimensional" (?)
from scipy import misc, ndimage
from scipy.fft import dctn, idctn 

# ------------------------------------------------------------- #
# "imaginea pe care o vom folosi la tema"
X = misc.ascent()
print(f" X[0,:] = {X[0,:]} ")
print( type(X) )
    # Q: "cmap" inseamna "color map" ??
    # Obs./Q: "metoda plt.imshow()" pt. imagini, in loc de "plt.plot()"
#plt.imshow(X, cmap = plt.cm.gray)
plt.imshow(X) 
plt.show()

    # Obs.: metoda numpy.shape( <array_numpy> )
print( np.shape(X) )
# ------------------------------------------------------------- #
# "Discrete Cosine transform --> "tipurile (1 sau 2 sau 3 sau 4)" ; "freq_db" ??ce inseamna??

    # Q: ce inseamna " "tipul" transformatei cosinus discrete" ?? <--> "type = 1/2/3/4"
Y1 = dctn(X, type=1)
Y2 = dctn(X, type=2)
Y3 = dctn(X, type=3)
Y4 = dctn(X, type=4)

freq_db_1 = 20 * np.log10( abs(Y1) )
freq_db_2 = 20 * np.log10( abs(Y2) )
freq_db_3 = 20 * np.log10( abs(Y3) )
freq_db_4 = 20 * np.log10( abs(Y4) )

    # Obs/Q: "smecherie cu "subplot()" " ??
plt.subplot(221).imshow(freq_db_1)
plt.subplot(222).imshow(freq_db_2)
plt.subplot(223).imshow(freq_db_3)
plt.subplot(224).imshow(freq_db_4)
plt.show()

# ------------------------------------------------------------- #
    # Down-Sampling ??
    # Q: "proprietatea compresiei energiei"??; "punem 0 pe "frecventele DCT-ului"?? incepand cu "bin-ul"?? k" ??
k = 120
Y_ziped = Y2.copy()
print( f"nonzero = { ( np.count_nonzero(Y_ziped) / 512**2 ) * 100}%" )
Y_ziped[k:] = 0
print( f"nonzero = { ( np.count_nonzero(Y_ziped) / 512**2 ) * 100}%" )
#print( f"nonzero = { (512**2 / np.count_nonzero(Y_ziped) ) * 100}%" )
#print( f"nonzero = {np.count_nonzero(Y_ziped)}" )
X_ziped = idctn(Y_ziped)
plt.imshow(X_ziped, cmap = plt.cm.gray)
plt.show()

# ------------------------------------------------------------- #
    # "Incercare de a face algo JPEG (fara ColorSpace Change)":

    # Matricea de Cuantizare Q
Q_jpeg = [[16, 11, 10, 16, 24, 40, 51, 61],
          [12, 12, 14, 19, 26, 28, 60, 55],
          [14, 13, 16, 24, 40, 57, 69, 56],
          [14, 17, 22, 29, 51, 87, 80, 62],
          [18, 22, 37, 56, 68, 109, 103, 77],
          [24, 35, 55, 64, 81, 104, 113, 92],
          [49, 64, 78, 87, 103, 121, 120, 101],
          [72, 92, 95, 98, 112, 100, 103, 99]]


"""
for x_ref in range(0, np.shape(X)[0] // 8 ):
    for y_ref in range(0, np.shape(X)[1] // 8 ):
        # "Selectam" fiecare block 8x8 din imagine:
        block = X[ x_ref:x_ref+8 ][ y_ref:y_ref+8 ]
"""

# "incercam sa facem DCT pe fiecare block "in place" i.e. fara sa facem copie la "matricea -- imaginea originala X"

X_2 = X.copy()
print( f"shape al X_2 = {np.shape(X_2)}" )

for x_ref in range(0, np.shape(X_2)[0] // 8 ):
    for y_ref in range(0, np.shape(X_2)[1] // 8 ):
        # pt. fiecare block facem "in place" DCT, apoi cuantizare

        #print(f"shape = {np.shape(X_2[ 8*x_ref:8*(x_ref+1) , 8*y_ref:8*(y_ref+1) ])}")
        #print(f"(x,x+8,y,y+8) = ({x_ref},{x_ref+8},{y_ref},{y_ref+8})")
        X_2[ 8*x_ref:8*(x_ref+1) , 8*y_ref:8*(y_ref+1) ] = Q_jpeg * np.round(  dctn( X_2[8*x_ref:8*(x_ref+1), 8*y_ref:8*(y_ref+1)] )  /  Q_jpeg  )
        
        # decodam ( i.e. facem inverse DCT ) (??)
            # Randul de mai jos e gresit pt. ca "parcurgerea matricei X_2 (cu x_ref si y_ref)" era facuta gresit !
            #X_2[ x_ref:x_ref+8 , y_ref:y_ref+8 ] = idctn( X_2[ x_ref:x_ref+8 , y_ref:y_ref+8 ] )
        X_2[ 8*x_ref:8*(x_ref+1) , 8*y_ref:8*(y_ref+1) ] = idctn( X_2[ 8*x_ref:8*(x_ref+1) , 8*y_ref:8*(y_ref+1) ] )


    # Plotam
#plt.imshow(X_2, cmap = plt.cm.gray)
plt.subplot(121).imshow(X_2, cmap = plt.cm.gray)
plt.subplot(122).imshow(X, cmap = plt.cm.gray)
plt.show()

"""
print( np.shape(  dctn( X_2[ 0:8 , 0:8 ] )  )  )
x = X_2[ 0:8 , 0:8 ]
print( np.shape(  dctn(x)  ) )
"""

# ------------------------------------------------------------- #
    # "Punem incercarea de mai sus intr-o fctie"
    # Test "ce se intampla daca comprimam de n ori imaginea"
def JPEG_Compress_Image(X, Q_jpeg):
    X_2 = X.copy()
    """
        # Procentajul original de intrari nonzero
    print(f" procentajul original de intrari nonzero = { np.count_nonzero(X_2) / ( np.shape(X_2)[0] * np.shape(X_2)[1] )  * 100}%")
    Eroare =  np.count_nonzero(X_2) / ( np.shape(X_2)[0] * np.shape(X_2)[1] )  * 100
    """

    for x_ref in range(0, np.shape(X_2)[0] // 8 ):
        for y_ref in range(0, np.shape(X_2)[1] // 8 ):
            # pt. fiecare block facem "in place" DCT, apoi cuantizare

            X_2[ 8*x_ref:8*(x_ref+1) , 8*y_ref:8*(y_ref+1) ] = Q_jpeg * np.round(  dctn( X_2[8*x_ref:8*(x_ref+1), 8*y_ref:8*(y_ref+1)] )  /  Q_jpeg  )
            
            # decodam ( i.e. facem inverse DCT ) (??)
            X_2[ 8*x_ref:8*(x_ref+1) , 8*y_ref:8*(y_ref+1) ] = idctn( X_2[ 8*x_ref:8*(x_ref+1) , 8*y_ref:8*(y_ref+1) ] )

    """
            # Procentajul de "intrari nonzero" de dupa Quantizare
    print(f" procentajul de intrari nonzero de dupa Q = { np.count_nonzero(X_2) / ( np.shape(X_2)[0] * np.shape(X_2)[1] ) * 100}%")
            # Eroare intre procentaje
    Eroare -=  np.count_nonzero(X_2) / ( np.shape(X_2)[0] * np.shape(X_2)[1] ) * 100
    print(f" Eroare intre procentaje = {Eroare}% \n" )
    """

    return X_2

"""
    # "Facem compresie de n ori pe ac. imagine"
n = 10
Img_JPEG = X.copy()
for i in range(0, n):
    print(f" Compresie de {i+1} ori: ")
    Img_JPEG = JPEG_Compress_Image(Img_JPEG, Q_jpeg)

plt.subplot(121).imshow(X, cmap = plt.cm.gray)
plt.subplot(122).imshow(Img_JPEG, cmap = plt.cm.gray)
plt.show()
"""
# ------------------------------------------------------------- #
    # "Cu procentaje de "intrari nonzero" "

def JPEG_Compress_Image_cuProcentaje(X, Q_jpeg):
    X_2 = X.copy()

    # Obs: Procentajele trebuiesc defapt "luate" pt. matricile "DCT-izate", pre si post Quantizare!! [nu pe matricile "non DCT-izate" (i.e. "cea initiala si cea "decodata" ")]
    """
            GRESIT!
        # Procentajul original de "Intrari nonzero"
    print(f" procentajul original de intrari nonzero = { np.count_nonzero(X_2) / ( np.shape(X_2)[0] * np.shape(X_2)[1] )  * 100}%")
    """

        # DCT pe block-uri 8x8
    for x_ref in range(0, np.shape(X_2)[0] // 8 ):
        for y_ref in range(0, np.shape(X_2)[1] // 8 ):
            X_2[ 8*x_ref:8*(x_ref+1) , 8*y_ref:8*(y_ref+1) ] = dctn( X_2[ 8*x_ref:8*(x_ref+1) , 8*y_ref:8*(y_ref+1) ] )
        # acum matricea X_2 e defapt X "DCT-izat"

        # Procentajul original de "Intrari nonzero"
    print(f" procentajul original de intrari nonzero (al matr. DCT-izate!) = { np.count_nonzero(X_2) / ( np.shape(X_2)[0] * np.shape(X_2)[1] )  * 100}%")
    print(f" numarul original de intrari nonzero (al matr. DCT-izate!) = {np.count_nonzero(X_2)}")

        # Facem "Quantizare"; => pierdere de informatie ; 
    for x_ref in range(0, np.shape(X_2)[0] // 8):
        for y_ref in range(0, np.shape(X_2)[1] // 8):
            #X_2[ 8*x_ref:8*(x_ref+1) , 8*y_ref:8*(y_ref+1) ] = idctn(  Q_jpeg * np.round( X_2[ 8*x_ref:8*(x_ref+1) , 8*y_ref:8*(y_ref+1) ] / Q_jpeg )  )

            X_2[ 8*x_ref:8*(x_ref+1) , 8*y_ref:8*(y_ref+1) ] = Q_jpeg * np.round(  X_2[8*x_ref:8*(x_ref+1), 8*y_ref:8*(y_ref+1)]  /  Q_jpeg  )  
        # acum matricea X_2 e "DCT-izata si Quantizata" si "intoarsa in domeniul RGB prin InverseDCT"
        # Procentajul de "intrari nonzero" de dupa Quantizare
    print(f" procentajul de intrari nonzero de dupa Q (al matr. DCT-izate!) = { np.count_nonzero(X_2) / ( np.shape(X_2)[0] * np.shape(X_2)[1] ) * 100}%")
    print(f" numarul de intrari nonzero dupa Q (al matr. DCT-izate!) = {np.count_nonzero(X_2)}\n")

        # Facem Inverse-DCT (ca sa ne intoarcem la formatul RGB)
    for x_ref in range(0, np.shape(X_2)[0] // 8 ):
        for y_ref in range(0, np.shape(X_2)[1] // 8 ):
            X_2[ 8*x_ref:8*(x_ref+1) , 8*y_ref:8*(y_ref+1) ] = idctn(  X_2[ 8*x_ref:8*(x_ref+1) , 8*y_ref:8*(y_ref+1) ]  )

    """
            GRESIT!
        # Procentajul de "intrari nonzero" de dupa Quantizare
    print(f" procentajul de intrari nonzero de dupa Q = { np.count_nonzero(X_2) / ( np.shape(X_2)[0] * np.shape(X_2)[1] ) * 100}%")
    """

    return X_2


    # "Aplicam JPEG_Compress_Image_cuProcentaje "
"""
X = misc.ascent()
Img_JPEG = X.copy()
Img_JPEG = JPEG_Compress_Image_cuProcentaje(X, Q_jpeg)
plt.subplot(121).imshow(X, cmap = plt.cm.gray)
plt.subplot(122).imshow(Img_JPEG, cmap = plt.cm.gray)
plt.show()
"""
    
    # Aplicam JPEG_Compress_Image_cuProcentaje de n ori pe ac. imagine
n = 10
Img_JPEG = X.copy()
for i in range(0, n):
    Img_JPEG = JPEG_Compress_Image_cuProcentaje(Img_JPEG, Q_jpeg)
plt.subplot(121).imshow(X, cmap = plt.cm.gray)
plt.subplot(121).set_title("Imaginea originala")
plt.subplot(122).imshow(Img_JPEG, cmap = plt.cm.gray)
plt.subplot(122).set_title(f"Imaginea comprimata cu \"Jpeg\" de {n} ori")
plt.show()




# ------------------------------------------------------------- #
