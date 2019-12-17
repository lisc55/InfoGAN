class FLAGS(object):
    def __init__(self):
        self.n_epoch = 100
        self.D_learning_rate = 2e-4
        self.G_learning_rate = 1e-3
        self.batch_size = 64
        self.leaky_rate = 0.1


flags = FLAGS()
