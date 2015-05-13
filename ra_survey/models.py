from utils import gauss_2d_anisotropic


class Model(object):
    """
    Basic class that implements models.

    :param x:
        Numpy array of coordinates with shape(npoints, ndim,).

    """
    def __init__(self, x, *args, **kwargs):
        self.x = x
        self.args = args
        self.kwargs = kwargs

    def __call__(self, p):
        raise NotImplementedError


class Model_2d_anisotropic(Model):
    def __call__(self, p):
        """
        :param p:
            Array-like parameter vector (log of amplitude [logJy], lg of
            brightness temperature [lgK], minor-to-major axis ratio, positional
            angle of major axis in image-plane).
        :return:
            Numpy array of values of the model with parameters represented by
            array-like ``p`` for points, specified in constructor.

        """
        u = self.x[:, 0]
        v = self.x[:, 1]
        return gauss_2d_anisotropic(p, u, v, *self.args, **self.kwargs)
