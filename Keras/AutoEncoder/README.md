# Autoencoder



![alt text](https://github.com/nakmuaycoder/start-with-MNIST/blob/master/img/autoencoder.jpg)


## Create an autoencoder model

Create a projection of mnist in a 16 dimensions space, then recreate it from the latent vector:
[jupyter notebook](https://github.com/nakmuaycoder/start-with-MNIST/blob/master/Keras/AutoEncoder/conv-autoencoder.ipynb)

![alt text](https://github.com/nakmuaycoder/start-with-MNIST/blob/master/img/autoenc_generated1.png)![alt text](https://github.com/nakmuaycoder/start-with-MNIST/blob/master/img/autoenc_generated2.png)

## Use an autoencoder to remove the noise of a mnist digit

Alter minst digits quality by adding noise, then train an autoencoder to regenerate the the original digit from the altered version.
[jupyter notebook](https://github.com/nakmuaycoder/start-with-MNIST/blob/master/Keras/AutoEncoder/DenoisingAutoEncoder.ipynb)

![alt text](https://github.com/nakmuaycoder/start-with-MNIST/blob/master/img/denoiser_generated1.png)
