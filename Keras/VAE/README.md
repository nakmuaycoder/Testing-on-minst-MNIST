# Variational Autoencoder



## About Variational Autoencoder [Pas fini]

### Network Architecture

A Variational Autoencoder <img src="https://render.githubusercontent.com/render/math?math=\mathcal{P}: \mathcal{D}\mapsto \mathcal{D}"> is similar to an autoencoder , and is made of an encoder <img src="https://render.githubusercontent.com/render/math?math=\mathcal{Q}: \mathcal{D} \mapsto \mathcal{Z}"> and a decoder <img src="https://render.githubusercontent.com/render/math?math=\mathcal{R}: \mathcal{Z} \mapsto \mathcal{D}. \forall x \in \mathcal{D},  x \sim \mathcal{P}(x) = \mathcal{R} \circ \mathcal{Q} (x)"> and a latent vector <img src="https://render.githubusercontent.com/render/math?math=z"> where  <img src="https://render.githubusercontent.com/render/math?math=\forall x \in \mathcal{D}, \exists z \in \mathcal{z} / \mathcal{Q} (x) = z">.

<img src="https://render.githubusercontent.com/render/math?math=\mathcal{P}_\theta (x) = \int_{\mathcal{Z}} \mathcal{P}_{\theta}(x,z) "> using Bayes theorem <img src="https://render.githubusercontent.com/render/math?math=\mathcal{P}_\theta (x) = \int_{\mathcal{Z}} \mathcal{P}_{\theta}(x | z)\mathcal{P}(z)=\int_{\mathcal{Z}} \mathcal{P}_{\theta}(z | x)\mathcal{P}(x)"><br>

<img src="https://render.githubusercontent.com/render/math?math=\mathcal{Q}(z|x)"> is chosen to be gaussian: <img src="https://render.githubusercontent.com/render/math?math=\mathcal{Q}(z|x) \sim \mathcal{N}(\mu(x),\sigma(x)\mathbb{I})">

The aim of the training is estimating <img src="https://render.githubusercontent.com/render/math?math=(\phi,\theta)"> where <img src="https://render.githubusercontent.com/render/math?math=\mathcal{Q}_{\phi}(z|x) \approx \mathcal{P}_{\theta}(z|x) ">


### Loss function

The loss function of a VAE has 2 parts:
- The reconstruction loss that measure the distance between the input and the output of the VAE ( MSE; Binary Cross Entropy)
- The Kullback-Leibler (KL) divergence measuring the distance between <img src="https://render.githubusercontent.com/render/math?math=\mathcal{Q}(z|x)"> and <img src="https://render.githubusercontent.com/render/math?math=\mathcal{P}(z|x)"> 


## Multilayer Perceptron Variational Autoencoder

![alt text](https://github.com/nakmuayFarang/start-with-MNIST/blob/master/img/vae-mlp.jpg)


### Convolutional Variational Autoencoder
![alt text](https://github.com/nakmuayFarang/start-with-MNIST/blob/master/img/vae_cnn.jpg)


### Conditional Variational Autoencoder
![alt text](https://github.com/nakmuayFarang/start-with-MNIST/blob/master/img/vae_cond.jpg)

