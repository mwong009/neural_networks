import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# Helper Functions
def zeros(shape):
    return np.zeros(shape, dtype=theano.config.floatX)

def ones(shape):
    return np.ones(shape, dtype=theano.config.floatX)

def glorot(rng, n_in, n_out):
    return np.asarray(
        rng.uniform(
            low=-4*np.sqrt(6./(n_in+n_out)),
            high=4*np.sqrt(6./(n_in+n_out)),
            size=(n_in, n_out)
        ),
        dtype=theano.config.floatX
    )

def uniform(rng, size, low=-1., high=1.):
    return np.asarray(
        rng.uniform(
            low=low,
            high=high,
            size=size
        ),
        dtype=theano.config.floatX
    )

def dropout(rng, layer, p):
    """ Perform the dropout function

    :type rng: numpy.random.RandomState
    :param rng: A random number generator

    :type layer: theano.tensor.TensorType
    :param layer: symbolic variable that describes the input

    :type p: float
    :param p: the probability that the neurons are deactivated
    """
    srng = T.shared_randomstreams.RandomStreams(rng.randint(1000))
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    output = layer * T.cast(mask, theano.config.floatX)

    return output / (1-p)
