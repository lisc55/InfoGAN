class FLAGS(object):
    def __init__(self):
        self.n_epoch = 100
        self.D_learning_rate = 2e-4
        self.G_learning_rate = 5e-4
        self.leaky_rate = 0.2
        self.c_dim = 5
        self.cont_lambda = 10.0
        self.disc_lambda = 1.0
        self.dim_noise = 128
        self.z_dim = 189
        self.batch_size = 64
        self.output_size = 32
        self.n_channel = 1
        self.n_samples = 10
        self.save_every_epoch = 6
        self.save_every_it = 50
        self.data_dir = "/data2/lishuchen/Faces"
        self.checkpoint_dir = "checkpoint"
        self.result_dir = "result"


flags = FLAGS()
