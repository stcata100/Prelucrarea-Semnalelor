import numpy as np
import matplotlib.pyplot as plt

from scipy import misc, ndimage
from scipy.fft import dctn, idctn

import skimage as ski

import math

# ---------------------------------------------------------------- #
#   Cerinta 1
# ---------------------------------------------------------------- #

    # Imaginea cu care vom lucra
X = misc.ascent()

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

    for x_ref in range(0, np.shape(X_2)[0] // 8 ):
        for y_ref in range(0, np.shape(X_2)[1] // 8 ):
            # DCT si Cuantizare, pe fiecare bloc 8x8
            X_2[ 8*x_ref:8*(x_ref+1) , 8*y_ref:8*(y_ref+1) ] = Q_jpeg * np.round(  dctn( X_2[8*x_ref:8*(x_ref+1), 8*y_ref:8*(y_ref+1)] )  /  Q_jpeg  )
            
            # Inverse DCT   
            X_2[ 8*x_ref:8*(x_ref+1) , 8*y_ref:8*(y_ref+1) ] = idctn( X_2[ 8*x_ref:8*(x_ref+1) , 8*y_ref:8*(y_ref+1) ] )

    return X_2

def JPEG_Compress_Image_cu_count_nonzero(X):
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

   # Plotare 
fig, axs = plt.subplots(1,2)
fig.suptitle("(Cerinta 1): ")

axs[0].imshow(X, cmap = plt.cm.gray)
axs[0].set_title("Imaginea originala")
axs[1].imshow(JPEG_Compress_Image_cu_count_nonzero(X), cmap = plt.cm.gray)
axs[1].set_title("Imaginea dupa comprimarea JPEG")
fig.savefig("Cerinta 1")
plt.show()

# ---------------------------------------------------------------- #
#   Cerinta 2
# ---------------------------------------------------------------- #

    # ---------------------------------------------------------------- #
    #   Trecerea de la un color-space la altul RGB <--> YCbCr
    # ---------------------------------------------------------------- #

    # Imagine RGB
Cat = ski.data.chelsea() 

    # RGB --> YCbCr
def RGB_to_YCbCr(Img):

    Img_copy = Img.copy()
            # Matricea de transformare RGB --> YCbCr
    M_rgb_to_ycbcr = np.array([[0.299, 0.587, 0.114],
                               [-0.167, -0.3313, 0.5],
                               [0.5, -0.4187, -0.0813]]).reshape( (3,3) ) 
    
    
    for i in range(0, np.shape(Img_copy)[0]):
        for j in range(0, np.shape(Img_copy)[1]):
            Img_copy[i, j, : ] = np.round(  ( M_rgb_to_ycbcr @ Img_copy[i, j, : ] ) + [0, 128, 128]  )

    return Img_copy

    # YCbCr --> RGB
def YCbCr_to_RGB(Img):

    Img_copy = Img.copy()
            # Matricea de transformare YCbCr --> RGB
    M_inversa = np.linalg.inv(  np.array([[0.299, 0.587, 0.114],
                               [-0.167, -0.3313, 0.5],
                               [0.5, -0.4187, -0.0813]]).reshape( (3,3) )  )
    
    for i in range(0, np.shape(Img_copy)[0]):
        for j in range(0, np.shape(Img_copy)[1]):
            # "Inmultirea matrice-vector"
            Img_copy[i, j, : ] = np.round(  M_inversa @ ( Img_copy[i, j, : ] - [0, 128, 128] )  )

    return Img_copy
    
    # ---------------------------------------------------------------- #
    # Exemplu pt. "trecerea de la un color-space la altul" (RGB <--> YCbCr)
    # ---------------------------------------------------------------- #
fig, axs = plt.subplots(3,4)
fig.suptitle("(Cerinta 2): Exemplu pt. \"trecerea de la un color-space la altul\" (RGB <--> YCbCr)")
        # Plotam "imaginea in color-space-ul RGB"
axs[0][0].imshow( Cat )
axs[0][0].set_title("Originala RGB")
axs[0][1].imshow( Cat[ : , : , 0] )
axs[0][1].set_title("R")
axs[0][2].imshow( Cat[ : , : , 1] )
axs[0][2].set_title("G")
axs[0][3].imshow( Cat[ : , : , 2] )
axs[0][3].set_title("B")

        # RGB --> YCbCr
Cat_ycbcr = RGB_to_YCbCr( Cat )
        # Plotam "imaginea in color-space-ul YCbCr"
axs[1][0].imshow( Cat_ycbcr )
axs[1][0].set_title("Originala $YC_bC_r$")
axs[1][1].imshow( Cat_ycbcr[ : , : , 0] )
axs[1][1].set_title("Luminance Y")
axs[1][2].imshow( Cat_ycbcr[ : , : , 1] )
axs[1][2].set_title("$C_b$")
axs[1][3].imshow( Cat_ycbcr[ : , : , 2] )
axs[1][3].set_title("$C_r$")

        # YC_bC_r --> RGB
Cat_intoarsa_RGB = YCbCr_to_RGB( Cat_ycbcr )
        # Plotam 
axs[2][0].imshow( Cat_intoarsa_RGB)
axs[2][0].set_title("Originala \"intoarsa in RGB\"") 
axs[2][1].imshow( Cat_intoarsa_RGB[ : , : , 0] )
axs[2][1].set_title("R")
axs[2][2].imshow( Cat_intoarsa_RGB[ : , : , 1] )
axs[2][2].set_title("G")
axs[2][3].imshow( Cat_intoarsa_RGB[ : , : , 2] )
axs[2][3].set_title("B")

fig.savefig("(Cerinta 2): Exemplu Schimbare Colorspace")
plt.show()
# ---------------------------------------------------------------- #
    
    # ---------------------------------------------------------------- #
    # Subsampling
    # ---------------------------------------------------------------- #

def Subsampling(M, block_size = 2):
    # Teorema impartirii cu rest: shape[0 sau 1] = 8 * cit + rest
        # Deci "cit"(defapt min{cit_shape0 , cit_shape1} !!) este "max size pe care il putem alege pt. "block-urile de sub-sampling" "
        #   pentru ca altfel, matricea sub-sampled va fi mai mica de 8x8, si nu vom putea face Cuantizare asupra ei
    M_shape = np.shape(M)
    max_block_size = min( [M_shape[0] // 8, M_shape[1] // 8] )
    if block_size > max_block_size:
        block_size = max_block_size
        # Daca nu s-ar putea sa facem quantizare pe matricea sub-sampled, nu mai facem subsampling
    if block_size == 0:
        return (M_shape, 0, M)

        # Matricea sub-sampled
            # math.ceil() pt. cazul in care M_shape[0 sau 1] nu e divizibil cu block_size ("mai raman margini")
    M_subsampled = np.zeros(  ( math.ceil(M_shape[0] / block_size) , math.ceil(M_shape[1] / block_size) )  )

    for i in range(0, M_shape[0], block_size):
        for j in range(0, M_shape[1], block_size):
            M_subsampled[i // block_size , j // block_size] = np.average(  M[ i:i+block_size , j:j+block_size ]  )

        # Returnam toate astea ca ulterior sa putem scala M_subsampled la M_shape
    return (M_shape, block_size, M_subsampled)

def Subsampling_Inverse(M_shape, block_size, M_subsampled):
        # Daca nu s-a facut Subsampling, nu se face nici Subsampling_Inverse
    if block_size == 0:
        return M_subsampled
        # Matricea M_subsampled scalata la M_shape
    M = np.zeros(M_shape)

        # Umplem block-urile din M cu valorile corespunzatoare din M_subsampled
            # np.full_like(<array>, <fill_value>) --> "intoarce un array de acelasi shape ca <array>, care are valoarea <fill_value> peste tot" 
    for i in range(0, M_shape[0], block_size):
        for j in range(0, M_shape[1], block_size):
            M[ i:i+block_size , j:j+block_size ] = np.full_like(  M[ i:i+block_size , j:j+block_size ], M_subsampled[ i // block_size , j // block_size ]  )

    return M

    # ---------------------------------------------------------------- #
    # Exemplu Subsampling
    # ---------------------------------------------------------------- #


fig, axs = plt.subplots(1,2)
fig.suptitle("(Cerinta2): Exemplu Subsampling")
axs[0].imshow(X)
axs[0].set_title("Imaginea Originala")
X_sub = Subsampling(X, block_size = 10)
axs[1].imshow( Subsampling_Inverse(X_sub[0], X_sub[1], X_sub[2]) )
axs[1].set_title("Imaginea Subsampled si re-scalata")
fig.savefig("(Cerinta 2): Exemplu Subsampling")
plt.show()


    # ---------------------------------------------------------------- #


    # ---------------------------------------------------------------- #
    # Compresie JPEG pt. imagine RGB (cu Subsampling)
    # ---------------------------------------------------------------- #
def JPEG_Compression_RGB(X, subsampling_block_size = 2):
   
        # Schimbam color-space-ul (RGB --> YCbCr)
    X_2 = RGB_to_YCbCr(X)
        # Sub-sample-uim Cb si Cr
    Cb_sub = Subsampling(X_2[1], subsampling_block_size)
    Cr_sub = Subsampling(X_2[2], subsampling_block_size)
            # Inlocuim Cb si Cr din X_2  cu variantele lor sub-sampled
    X_2[2] = Cb_sub[2].copy()
    X_2[2] = Cr_sub[2].copy()
        # DCT si Quantizare asupra fiecareia dintre Y, Cb, Cr
    X_2[0] = JPEG_Compress_Image(X_2[0])
    X_2[1] = JPEG_Compress_Image(X_2[1])
    X_2[2] = JPEG_Compress_Image(X_2[2])

        # "Inversam sub-sample-uirea"
    X_2[1] = Subsampling_Inverse(Cb_sub[0], Cb_sub[1], Cb_sub[2])
    X_2[2] = Subsampling_Inverse(Cr_sub[0], Cr_sub[1], Cr_sub[2])
        # Schimbam color-space-ul (YCbCr --> RGB)
    X_2 = YCbCr_to_RGB(X_2)

    return X_2

    # Plotam
fig, axs = plt.subplots(1,2)
fig.suptitle("(Cerinta2)")
axs[0].imshow(Cat)
axs[0].set_title("Imaginea Originala")
axs[1].imshow( JPEG_Compression_RGB(Cat, subsampling_block_size = 1000) )
axs[1].set_title("Imaginea dupa JPEG_RGB_cu_subsampling")
fig.savefig("Cerinta2")
plt.show()

    # ---------------------------------------------------------------- #

# ---------------------------------------------------------------- #
# Cerinta 3
# ---------------------------------------------------------------- #

    # Presupunand ca nu schimbam Matricea de cuantizare, putem creste compresia crescand marimea block-urilor de la Subsampling(in aceste 2 etape se pierd date si creste MSE); cautam block_size optim a.i. MSE-ul(intre Imaginea originala si Imaginea comprimata JPEG) sa fie <= pragul impus de utilizator
def MSE_a_doua_imagini(X1, X2):
    M = X2 - X1
    M = M * M
    return np.average(M)
"""
#print(f"MSE = { MSE_a_doua_imagini( X, JPEG_Compress_Image(X) )}")
print(f"MSE = { MSE_a_doua_imagini( Cat, JPEG_Compression_RGB(Cat, 2) )}")
print(f"MSE = { MSE_a_doua_imagini( Cat, JPEG_Compression_RGB(Cat, 10) )}")
print(f"MSE = { MSE_a_doua_imagini( Cat, JPEG_Compression_RGB(Cat, 100) )}")
print(f"MSE = { MSE_a_doua_imagini( Cat, JPEG_Compression_RGB(Cat, 1000) )}")
print(f"MSE = { MSE_a_doua_imagini( Cat, JPEG_Compression_RGB(Cat, 10000) )}")
print(np.shape(X))
print(np.shape(JPEG_Compress_Image(X)))
"""

