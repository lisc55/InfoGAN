class FLAGS(object):
    def __init__(self):
        self.n_epoch = 100
        self.D_learning_rate = 2e-4
        self.G_learning_rate = 5e-4
        self.leaky_rate = 0.2
        self.batch_size = 64
        self.output_size = 32
        self.model_dir = "models/"
        self.res_dir = "results/"


flags = FLAGS()
