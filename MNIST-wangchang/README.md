## MNIST训练结果展示

下图示出了改变第1个连续latent code得到的结果. 可以看出网络对该code的理解是**控制数字的倾斜度**.

<img src="results/c1_res.png">

下图示出了改变第2个连续latent code得到的结果. 可以看出网络对该code的理解是**控制数字笔画的粗细**.

<img src="results/c2_res.png">

下图示出了改变离散latent code向量得到的结果. 可以看出网络对该向量的理解是**控制数字的种类**.

<img src="results/cat_res.png">

下面两张图分别为随机生成结果和训练过程中的loss变化曲线图.

<img src="results/random.png">

<img src="results/loss.png">