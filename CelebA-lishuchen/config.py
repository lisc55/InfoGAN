class FLAGS(object):
	def __init__(self):
		self.n_epoch = 25
		self.D_learning_rate = 0.0002
		self.G_learning_rate = 0.001
		self.leaky_rate = 0.1
		self.batch_size = 64
		self.output_size = 32
		self.n_channel = 3
		self.save_every_epoch = 1
		self.data_dir = "data"
		self.checkpoint_dir = "checkpoint"
flags = FLAGS()