class FLAGS(object):
	def __init__(self):
		self.n_epoch = 100
		self.D_learning_rate = 0.0002
		self.G_learning_rate = 0.0002
		self.beta_1 = 0.5
		self.leaky_rate = 0.1
		self.n_categorical = 10
		self.dim_categorical = 10
		self.dim_noise = 128
		self.dim_z = 228
		self.batch_size = 128
		self.output_size = 32
		self.n_channel = 3
		self.save_every_epoch = 1
		self.save_every_it = 10
		self.n_sample = 5
		self.data_dir = "CelebA"
		self.checkpoint_dir = "checkpoint"
		self.result_dir = "result"
flags = FLAGS()