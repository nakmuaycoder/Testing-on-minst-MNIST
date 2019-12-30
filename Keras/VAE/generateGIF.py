

import os
from glob import glob
import imageio

path = "D:/Project/DeepLearning/minst_VAE/Output/"

os.chdir(path)
for digit in range(9):
    img = list()
    for beta in [1,5,8,20]:
        img.append(imageio.imread('generatedDigits{}_beta{}.0.png'.format(digit,beta))  )
    imageio.mimsave("digit{}.gif".format(digit),img)

img = list()
for beta in [1,5,8,20]:
    img.append(  imageio.imread("latentSpace_beta{}.0.png".format(beta))  )
imageio.mimsave("LatentSpace.gif",img)