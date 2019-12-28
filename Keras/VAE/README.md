# Variational Autoencoder [A reprendre et finir]



## About Variational Autoencoder

This model is as an autoencoder, have an encoder <img src="https://render.githubusercontent.com/render/math?math=\mathcal{Q}: \mathcal{D} \mapsto \mathcal{Z}"> and a decoder <img src="https://render.githubusercontent.com/render/math?math=\mathcal{P}: \mathcal{Z} \mapsto \mathcal{D}"> and  <img src="https://render.githubusercontent.com/render/math?math=\forall x \in \mathcal{D},  x \sim \mathcal{P} \circ \mathcal{Q} (x)"> <br>

<img src="https://render.githubusercontent.com/render/math?math=\forall x \in \mathcal{D},  x \sim \mathcal{P} \circ \mathcal{Q} (x)"> 

and an encoder 
<img src="https://render.githubusercontent.com/render/math?math=\mathcal{Q}">  where  <img src="https://render.githubusercontent.com/render/math?math=\forall x \in \mathcal{D}, \exists z \in \mathcal{z} / \mathcal{Q} (x) = z">

<img src="https://render.githubusercontent.com/render/math?math=\mathcal{P}_\theta (x) = \int_{z} \mathcal{P}_{\theta}(x,z)dz "> using Bayes theorem <img src="https://render.githubusercontent.com/render/math?math=\mathcal{P}_\theta (x) = \int_{z} \mathcal{P}_{\theta}(x | z)\mathcal{P}(z)dz  ">



## Multilayer Perceptron Variational Autoencoder

![alt text](https://github.com/nakmuayFarang/start-with-MNIST/blob/master/img/vae-mlp.jpg)


### Convolutional Variational Autoencoder
![alt text](https://github.com/nakmuayFarang/start-with-MNIST/blob/master/img/vae_cnn.jpg)


### Conditional Variational Autoencoder

