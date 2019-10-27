import tensorflow as tf

def l2normalize(v, eps=1e-12):
    return tf.math.divide(v,(tf.norm(v) + eps))

class SpectralNormalization(tf.keras.layers.Layer):
    """ Paper: https://openreview.net/forum?id=B1QRgziT-
        source: https://github.com/pfnet-research/sngan_projection
    """

    def __init__(self, module, name="weights", Ip=1, factor=None):
        super(SpectralNormalization, self).__init__()
        self.module = module
        self.weight_name = name

        if not Ip >= 1:
            raise ValueError("The number of power iterations should be positive integer")
        self.Ip = Ip
        self.factor = factor

    def _check_param(self):
        try:
            u = getattr(self, "u")
            v = getattr(self, "v")
            return True
        except AttributeError:
            return False

    def _make_param(self):
        w = getattr(self.module, self.weight_name)[0]
        height = w.shape[-1]
        width = tf.reshape(w, shape=(height, -1)).shape[1]
        # print("H: ", height, "W: ", width)
        u = tf.random.normal(shape=[1, height])
        v = tf.random.normal(shape=[1, width])
        self.u = l2normalize(u)
        self.v = l2normalize(v)

    def build(self, input_shape):
        self.module.build(input_shape)
        if not self._check_param():
            self._make_param()
        
    def call(self, x, training=False):
        self.update_uv()
        return self.module.call(x)

    @tf.function
    def update_uv(self):
        """
        Spectrally Normalized Weight
        """
        W = getattr(self.module, self.weight_name)[0]
        W_mat = tf.reshape(W, [W.shape[-1], -1])

        for _ in range(self.Ip):
            self.v = l2normalize(tf.matmul(self.u, W_mat))
            self.u = l2normalize(tf.matmul(self.v, tf.transpose(W_mat)))
            
        sigma = tf.reduce_sum(tf.matmul(self.u, W_mat) * self.v)


        if self.factor:
            sigma = sigma / self.factor

        W =  W / sigma

