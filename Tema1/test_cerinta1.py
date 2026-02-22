import numpy as np
import matplotlib.pyplot as plt

from scipy import misc, ndimage
from scipy.fft import dctn, idctn 

X = misc.ascent()
"""
    # Test pt. ce scrie in notebook despre "Proprietatea de compresie"
A = np.array( [i for i in range(1,17)] ).reshape( (4,4) )
print(A)
print()
print(A[ 2: , : ])

    # Test "Varianta mai buna de "parcurgere pe block-uri" a Matricei"
        # Obs.(!!): "Ia si "bucatile din matrice" care nu sunt patratice !!
B = np.array( [i for i in range(1,26)] ).reshape( (5,5) )
for i in range(0, np.shape(B)[0], 2):
    for j in range(0, np.shape(B)[1], 2):
        print( B[ i:i+2 , j:j+2 ] )
        print()
"""


def JPEG_Compress_Image(X):
        # Matricea de cuantizare
    Q_jpeg = [[16, 11, 10, 16, 24, 40, 51, 61],
              [12, 12, 14, 19, 26, 28, 60, 55],
              [14, 13, 16, 24, 40, 57, 69, 56],
              [14, 17, 22, 29, 51, 87, 80, 62],
              [18, 22, 37, 56, 68, 109, 103, 77],
              [24, 35, 55, 64, 81, 104, 113, 92],
              [49, 64, 78, 87, 103, 121, 120, 101],
              [72, 92, 95, 98, 112, 100, 103, 99]]

    X_2 = X.copy()

    for i in range(0, np.shape(X_2)[0], 8):
        for j in range(0, np.shape(X_2)[1], 8):
                # DCT si Cuantizare
            X_2[ i:i+8 , j:j+8 ] = Q_jpeg * np.round( dctn( X_2[ i:i+8 , j:j+8 ] )  /  Q_jpeg )
                # Inverse DCT
            X_2[ i:i+8 , j:j+8 ] = idctn( X_2[ i:i+8 , j:j+8 ] )

    return X_2
    

def JPEG_Compress_Image_v2(X, nr_cuantizari = 1):
        # Matricea de cuantizare
    Q_jpeg = [[16, 11, 10, 16, 24, 40, 51, 61],
              [12, 12, 14, 19, 26, 28, 60, 55],
              [14, 13, 16, 24, 40, 57, 69, 56],
              [14, 17, 22, 29, 51, 87, 80, 62],
              [18, 22, 37, 56, 68, 109, 103, 77],
              [24, 35, 55, 64, 81, 104, 113, 92],
              [49, 64, 78, 87, 103, 121, 120, 101],
              [72, 92, 95, 98, 112, 100, 103, 99]]

    X_2 = X.copy()


    for i in range(0, np.shape(X_2)[0], 8):
        for j in range(0, np.shape(X_2)[1], 8):
                # DCT 
            X_2[ i:i+8 , j:j+8 ] = dctn( X_2[ i:i+8 , j:j+8 ] )
                    # Numaram cate intrari nonzero au fost inainte si dupa cuantizare
                # Cuantizare
            for k in range(0, nr_cuantizari):
                X_2[ i:i+8 , j:j+8 ] = Q_jpeg * np.round( X_2[ i:i+8 , j:j+8 ]  /  Q_jpeg )
            nr_nonzero_dupa = np.count_nonzero( X_2 )
                # Inverse DCT
            X_2[ i:i+8 , j:j+8 ] = idctn( X_2[ i:i+8 , j:j+8 ] )


    return X_2
    

def JPEG_Compress_Image_v3(X, nr_cuantizari = 1):
        # Matricea de cuantizare
    Q_jpeg = [[16, 11, 10, 16, 24, 40, 51, 61],
              [12, 12, 14, 19, 26, 28, 60, 55],
              [14, 13, 16, 24, 40, 57, 69, 56],
              [14, 17, 22, 29, 51, 87, 80, 62],
              [18, 22, 37, 56, 68, 109, 103, 77],
              [24, 35, 55, 64, 81, 104, 113, 92],
              [49, 64, 78, 87, 103, 121, 120, 101],
              [72, 92, 95, 98, 112, 100, 103, 99]]

    X_2 = X.copy()


                # DCT 
    for i in range(0, np.shape(X_2)[0], 8):
        for j in range(0, np.shape(X_2)[1], 8):
            X_2[ i:i+8 , j:j+8 ] = dctn( X_2[ i:i+8 , j:j+8 ] )
                    # Numaram cate intrari nonzero au fost inainte si dupa cuantizare
    nr_nonzero_inainte = np.count_nonzero(X_2)
                # Cuantizare
    for i in range(0, np.shape(X_2)[0], 8):
        for j in range(0, np.shape(X_2)[1], 8):
            for k in range(0, nr_cuantizari):
                X_2[ i:i+8 , j:j+8 ] = Q_jpeg * np.round( X_2[ i:i+8 , j:j+8 ]  /  Q_jpeg )
    nr_nonzero_dupa = np.count_nonzero(X_2)
                # Inverse DCT
    for i in range(0, np.shape(X_2)[0], 8):
        for j in range(0, np.shape(X_2)[1], 8):
            X_2[ i:i+8 , j:j+8 ] = idctn( X_2[ i:i+8 , j:j+8 ] )

    print(f"nr. intrari nonzero pre-Q = {nr_nonzero_inainte}")
    print(f"nr. intrari nonzero dupa Q = {nr_nonzero_dupa}")

    return X_2

def JPEG_Compress_Image_v4(X):
        # Matricea de cuantizare
    Q_jpeg = [[16, 11, 10, 16, 24, 40, 51, 61],
              [12, 12, 14, 19, 26, 28, 60, 55],
              [14, 13, 16, 24, 40, 57, 69, 56],
              [14, 17, 22, 29, 51, 87, 80, 62],
              [18, 22, 37, 56, 68, 109, 103, 77],
              [24, 35, 55, 64, 81, 104, 113, 92],
              [49, 64, 78, 87, 103, 121, 120, 101],
              [72, 92, 95, 98, 112, 100, 103, 99]]

    X_2 = X.copy()


                # DCT 
    for i in range(0, np.shape(X_2)[0], 8):
        for j in range(0, np.shape(X_2)[1], 8):
            X_2[ i:i+8 , j:j+8 ] = dctn( X_2[ i:i+8 , j:j+8 ] )
                    # Numaram cate intrari nonzero au fost inainte si dupa cuantizare
    nr_nonzero_inainte = np.count_nonzero(X_2)
                # Cuantizare
    for i in range(0, np.shape(X_2)[0], 8):
        for j in range(0, np.shape(X_2)[1], 8):
            X_2[ i:i+8 , j:j+8 ] = Q_jpeg * np.round( X_2[ i:i+8 , j:j+8 ]  /  Q_jpeg )
    nr_nonzero_dupa = np.count_nonzero(X_2)
                # Inverse DCT
    for i in range(0, np.shape(X_2)[0], 8):
        for j in range(0, np.shape(X_2)[1], 8):
            X_2[ i:i+8 , j:j+8 ] = idctn( X_2[ i:i+8 , j:j+8 ] )

    print(f"nr. intrari nonzero pre-Q = {nr_nonzero_inainte}")
    print(f"nr. intrari nonzero dupa Q = {nr_nonzero_dupa}")

    return X_2

    # Plotam imaginea originala
plt.subplot(121).imshow(X, cmap = plt.cm.gray)
plt.subplot(121).set_title("Imaginea originala")
"""
    # Aplicam de n ori JPEG_Compress_Image
Img_JPEG = X.copy()
n = 50
for i in range(0, n):
    Img_JPEG = JPEG_Compress_Image( Img_JPEG )
    # Plotam "imaginea comprimata"
plt.subplot(122).imshow(Img_JPEG, cmap = plt.cm.gray)
plt.subplot(122).set_title(f"Imaginea comprimata de {n} ori")
plt.show()
"""
Img_JPEG = X.copy()
n = 50 

for i in range(0, n):
    Img_JPEG = JPEG_Compress_Image_v4(Img_JPEG)
    # Obs.: "cresterea numarului de Quantizari nu face sa fie mai putine intrari non-zero dupa Quantizare (dupa cele n Quantizari)"
    # JPEG_Compress_Image_v4  e "cea mai imbunatatita" varianta (?)

plt.subplot(122).imshow(Img_JPEG, cmap = plt.cm.gray)
plt.subplot(122).set_title(f"Imaginea comprimata de {n} ori")
plt.show()

# ------------------------------------------------------------- #
"""
def JPEG_Compress_Image_v2(X, nr_cuantizari = 1):
        # Matricea de cuantizare
    Q_jpeg = [[16, 11, 10, 16, 24, 40, 51, 61],
              [12, 12, 14, 19, 26, 28, 60, 55],
              [14, 13, 16, 24, 40, 57, 69, 56],
              [14, 17, 22, 29, 51, 87, 80, 62],
              [18, 22, 37, 56, 68, 109, 103, 77],
              [24, 35, 55, 64, 81, 104, 113, 92],
              [49, 64, 78, 87, 103, 121, 120, 101],
              [72, 92, 95, 98, 112, 100, 103, 99]]

    X_2 = X.copy()

    for i in range(0, np.shape(X_2)[0], 8):
        for j in range(0, np.shape(X_2)[1], 8):
                # DCT 
            X_2[ i:i+8 , j:j+8 ] = dctn( X_2[ i:i+8 , j:j+8 ] )
                # Cuantizare
            for k in range(0, nr_cuantizari):
                X_2[ i:i+8 , j:j+8 ] = Q_jpeg * np.round( X_2[ i:i+8 , j:j+8 ]  /  Q_jpeg )
                # Inverse DCT
            X_2[ i:i+8 , j:j+8 ] = idctn( X_2[ i:i+8 , j:j+8 ] )

    return X_2
    
"""
def JPEG_Compress_Image_v4(X):
        # Matricea de cuantizare
    Q_jpeg = [[16, 11, 10, 16, 24, 40, 51, 61],
              [12, 12, 14, 19, 26, 28, 60, 55],
              [14, 13, 16, 24, 40, 57, 69, 56],
              [14, 17, 22, 29, 51, 87, 80, 62],
              [18, 22, 37, 56, 68, 109, 103, 77],
              [24, 35, 55, 64, 81, 104, 113, 92],
              [49, 64, 78, 87, 103, 121, 120, 101],
              [72, 92, 95, 98, 112, 100, 103, 99]]

    X_2 = X.copy()


                # DCT 
    for i in range(0, np.shape(X_2)[0], 8):
        for j in range(0, np.shape(X_2)[1], 8):
            X_2[ i:i+8 , j:j+8 ] = dctn( X_2[ i:i+8 , j:j+8 ] )
                    # Numaram cate intrari nonzero au fost inainte si dupa cuantizare
    nr_nonzero_inainte = np.count_nonzero(X_2)
                # Cuantizare
    for i in range(0, np.shape(X_2)[0], 8):
        for j in range(0, np.shape(X_2)[1], 8):
            X_2[ i:i+8 , j:j+8 ] = Q_jpeg * np.round( X_2[ i:i+8 , j:j+8 ]  /  Q_jpeg )
    nr_nonzero_dupa = np.count_nonzero(X_2)
                # Inverse DCT
    for i in range(0, np.shape(X_2)[0], 8):
        for j in range(0, np.shape(X_2)[1], 8):
            X_2[ i:i+8 , j:j+8 ] = idctn( X_2[ i:i+8 , j:j+8 ] )

    print(f"nr. intrari nonzero pre-Q = {nr_nonzero_inainte}")
    print(f"nr. intrari nonzero dupa Q = {nr_nonzero_dupa}")

    return x_2
