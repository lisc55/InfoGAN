# InfoGAN
## InfoGAN Architecture 

Tensorlayer implementation of [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://arxiv.org/abs/1606.03657).

<div align="center">
	<img src='img/architecture.svg' width="60%" height="50%">
</div>

## Results

### MNIST

#### Manipulating the First Continuous Latent Code

```diff
- Changing c1 will rotate the digits.
```

<div align="center">
	<img src='./MNIST-wangchang/results/c1_res.png' width="100%">
</div>

#### Manipulating the Second Continuous Latent Code

```diff
- Changing c2 will change the width of the digits.
```

<div align="center">
	<img src='./MNIST-wangchang/results/c2_res.png' width="100%">
</div>

#### Manipulating the Discrete Latent Code (Categorical)

```diff
- Changing d will change the type of digits
```

<div align="center">
	<img src='./MNIST-wangchang/results/cat_res.png' width="100%">
</div>

#### Random Generation and Loss Plot

<div align="center">
	<img src='./MNIST-wangchang/results/random.png' width="100%">
</div>

G_loss increases steadily after a sufficient number of iterations, showing the discriminator is getting stronger and stronger and indicating the end of training.

<div align="center">
	<img src='./MNIST-wangchang/results/loss.png' width="100%">
</div>

### CelebA

#### Manipulating Discrete Latent Code

Azimuth (pose):

<div align="center">
	<img src='./CelebA-lishuchen/samples/Azimuth.png' width="80%" height="50%">
</div>

Presence or absence of glasses:

<div align="center">
	<img src='./CelebA-lishuchen/samples/Glasses.png' width="80%" height="50%">
</div>

Hair color:

<div align="center">
	<img src='./CelebA-lishuchen/samples/Hair_color.png' width="80%" height="50%">
</div>

Hair quantity:

<div align="center">
	<img src='./CelebA-lishuchen/samples/Hair_quantity.png' width="80%" height="50%">
</div>

Lighting:

<div align="center">
	<img src='./CelebA-lishuchen/samples/Lighting.png' width="80%" height="50%">
</div>

## Run

#### MNIST

* Start training using ```python train.py```; this will automatically download the dataset.
* To see the results, execute ```python test.py``` and **input the number of your saved model**.
* Feel free to manipulate the parameters in ```test.py```.

#### CelebA

+ Set your image folder in `config.py`.
+ Some links for the datasets:
	+ [CelebA](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8)
+ Start training.

```
python train.py
```

## References

1. [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://arxiv.org/abs/1606.03657)
2. [Large-scale CelebFaces Attributes (CelebA) Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

## Authors

+ [李舒辰 (@lisc55)](https://github.com/lisc55): The experiment on CelebA.
+ [王畅 (@wangchang327)](https://github.com/wangchang327): The experiment on MNIST.

