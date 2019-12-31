# InfoGAN
###InfoGAN Architecture 

Tensorlayer implementation of ["InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets"](https://arxiv.org/abs/1606.03657).

<img src='img/architecture.pdf' width="80%" height="50%">

### Results

#### CelebA

##### Manipulating Discrete Latent Code

Azimuth (pose):

<img src='./CelebA-lishuchen/samples/Azimuth.png' width="100%" height="50%">

Presence or absence of glasses:

<img src='./CelebA-lishuchen/samples/Glasses.png' width="100%" height="50%">

Hair color:

<img src='./CelebA-lishuchen/samples/Hair_color.png' width="100%" height="50%">

Hair quantity:

<img src='./CelebA-lishuchen/samples/Hair_quantity.png' width="100%" height="50%">

Lighting:

<img src='./CelebA-lishuchen/samples/Lighting.png' width="100%" height="50%">

### Run

+ Set your image folder in `config.py`.
+ Some links for the datasets:
	+ [CelebA](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8)
+ Start training.

```
python train.py
```

### Reference

1. ["InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets"](https://arxiv.org/abs/1606.03657)
2. [Large-scale CelebFaces Attributes (CelebA) Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

### Author

+ [李舒辰 (@lisc55)](https://github.com/lisc55)

