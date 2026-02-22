import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
    # Q: "ce inseamna astea de mai jos"??
        # dctn := "discrete cosine transform -- n dimensional" (?)
        # idctn := "INVERSE discrete cosine transform -- n dimensional" (?)
from scipy import misc, ndimage
from scipy.fft import dctn, idctn 

    # Chestie noua ca sa importam Imagine de test cu pisica -- RGB !!
        # ski.data.cat()  ( sau ski.data.chelsea() )
import skimage as ski

    # Imagine de test cu pisica; care ar trebui sa fie RGB!! (nu Grayscale)
Cat = ski.data.chelsea()
print(f" shape(Cat) = {np.shape(Cat)}")
#Cat = ski.data.astronaut()
plt.subplot(111).imshow(Cat)
plt.subplot(111).set_title("Chelsea the cat :) ")
plt.show()

    # Plotam RGB
plt.subplot(131).imshow( Cat[ : , : , 0 ] )
plt.subplot(131).set_title("Red")
plt.subplot(132).imshow( Cat[ : , : , 1 ] )
plt.subplot(132).set_title("Green")
plt.subplot(133).imshow( Cat[ : , : , 2 ] )
plt.subplot(133).set_title("Blue")
plt.show()

#list(colormaps)

    # Convertire colorspace folosind functia built-in din "scikit-image"
Cat_ycbcr = ski.color.convert_colorspace(Cat, "RGB", "YCbCr")

    # Plotam YCbCr
plt.subplot(231).imshow( Cat_ycbcr[ : , : , 0 ] )
plt.subplot(231).set_title("Luminance Y")
plt.subplot(232).imshow( Cat_ycbcr[ : , : , 1 ] )
plt.subplot(232).set_title("$C_b$")
plt.subplot(233).imshow( Cat_ycbcr[ : , : , 2 ] )
plt.subplot(233).set_title("$C_r$")
plt.show()
"""
plt.subplot(111).imshow( Cat_ycbcr )
plt.subplot(111).set_title("Cat $YC_bC_r$")
plt.show()
"""

#list(colormaps)

    # ------------------------------------------- #
    # Convertire "manuala" RGB --> YCbCr


def RGB_to_YCbCr(Img):

    Img_copy = Img.copy()
            # Matricea de transformare RGB --> YCbCr
    M_rgb_to_ycbcr = np.array([[0.299, 0.587, 0.114],
                               [-0.167, -0.3313, 0.5],
                               [0.5, -0.4187, -0.0813]]).reshape( (3,3) ) 
    
    
    for i in range(0, np.shape(Img_copy)[0]):
        for j in range(0, np.shape(Img_copy)[1]):
            # "Inmultirea matrice-vector"
            #Img_copy[i, j, : ] = ( M_rgb_to_ycbcr @ Img_copy[i, j, : ].reshape( (3,1) ) ).reshape(3) + [0, 128, 128]
            Img_copy[i, j, : ] = np.round(  ( M_rgb_to_ycbcr @ Img_copy[i, j, : ] ) + [0, 128, 128]  )

    return Img_copy

    # Schimbarea manuala a color-space-ului
Cat_ycbcr_manual = RGB_to_YCbCr(Cat)
    
    # Plotam
plt.subplot(151).imshow( Cat_ycbcr_manual[ : , : , 0 ] )
plt.subplot(151).set_title("Luminance Y (manual)")
plt.subplot(152).imshow( Cat_ycbcr_manual[ : , : , 1 ] )
plt.subplot(152).set_title("$C_b$ (manual)")
plt.subplot(153).imshow( Cat_ycbcr_manual[ : , : , 2 ] )
plt.subplot(153).set_title("$C_r$ (manual)")
plt.subplot(154).imshow( Cat_ycbcr_manual )
plt.subplot(154).set_title("Originala $YC_bC_r$ (manual)")
plt.subplot(155).imshow( Cat )
plt.subplot(155).set_title("Originala $RGB$ (manual)")
plt.show()

    
print( Cat_ycbcr_manual )



    # ------------------------------------------- #
        # YCbCr_to_RGB

def YCbCr_to_RGB(Img):

    Img_copy = Img.copy()
            # Matricea de transformare YCbCr --> RGB
    M_inversa = np.linalg.inv(  np.array([[0.299, 0.587, 0.114],
                               [-0.167, -0.3313, 0.5],
                               [0.5, -0.4187, -0.0813]]).reshape( (3,3) )  )
    
    for i in range(0, np.shape(Img_copy)[0]):
        for j in range(0, np.shape(Img_copy)[1]):
            # "Inmultirea matrice-vector"
            #Img_copy[i, j, : ] = (M_inversa @ ( Img_copy[i, j, : ] - [0, 128, 128] ).reshape( (3,1) ) ).reshape(3) 
            Img_copy[i, j, : ] = np.round(  M_inversa @ ( Img_copy[i, j, : ] - [0, 128, 128] )  )

    return Img_copy

plt.subplot(111).imshow( YCbCr_to_RGB(Cat_ycbcr_manual) )
plt.subplot(111).set_title("Inversarea trecerii manuale")
plt.show()
#print(f" Max al cat_ycbcr = {np.max(Cat_ycbcr_manual)}")

#print( f" Shape of Cat[0,0, : ] = {np.shape( Cat[0,0, : ] )}" )
    # ------------------------------------------- #





