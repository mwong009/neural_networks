import pickle, h5py, gzip
import os, sys, timeit

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from optimizers import *
from helpers import *

class Layer(object):
    """ Generic artificial neural network container class """
    def __init__(self, input, output, n_in, n_out):
        """ Initialize parameters of the layer

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input

        :type n_in: int
        :param n_in: number of input units

        :type n_out: int
        :param n_out: number of output units
        """

        # keeping track of the model
        self.input = input
        self.output = output
        self.n_in = n_in
        self.n_out = n_out
        self.params = []

class Network(object):
    """ Network class container """
    def __init__(self, input, output, n_in, n_out, rng, optimizer):
        """ Initialize parameters of the layer

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input

        :type n_in: int
        :param n_in: number of input units

        :type n_out: int
        :param n_out: number of output units
        """

        # keeping track of the model
        self.input = input
        self.output = output
        self.n_in = n_in
        self.n_out = n_out
        self.optimizer = optimizer
        self.cost = None
        self.params = []

        if rng is None:
            rng = np.random.RandomState(1234)

        self.rng = rng
        self.learning_rate = None
        self.momentum = None

    def build_fns(self, dataset, batch_size, learning_rate, momentum):
        """Generates theano functions 'train_fn', 'valid_fn' and 'test_fn' """

        train_set_x, train_set_y = dataset[0]
        valid_set_x, valid_set_y = dataset[1]
        test_set_x, test_set_y = dataset[2]
        # compute number of minibatches for training, validation and testing
        n_train_samples = train_set_x.get_value(borrow=True).shape[0]
        n_train_batches = n_train_samples // batch_size
        n_valid_samples = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches = n_valid_samples // batch_size
        n_test_samples = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches = n_test_samples // batch_size

        print('... building the model')
        index = T.lscalar('index') # index to a [mini]batch

        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size

        self.learning_rate = learning_rate
        self.momentum = momentum

        # compute the gradients with respect to the model parameters
        self.gparams = T.grad(self.cost, self.params)
        opt = self.optimizer(self.params)
        updates = opt.updates(self.params, self.gparams,
            self.learning_rate, self.momentum)

        train_fn = theano.function(
            inputs=[index],
            outputs=self.cost,
            updates=updates,
            givens={
                self.input: train_set_x[batch_begin: T.switch(T.eq(index, n_train_batches-1), -1, batch_begin + batch_size)],
                self.output: train_set_y[batch_begin: T.switch(T.eq(index, n_train_batches-1), -1, batch_begin + batch_size)]
            }
        )

        valid_fn = theano.function(
            inputs=[index],
            outputs=self.errors,
            givens={
                self.input: valid_set_x[batch_begin: T.switch(T.eq(index, n_valid_batches-1), -1, batch_begin + batch_size)],
                self.output: valid_set_y[batch_begin: T.switch(T.eq(index, n_valid_batches-1), -1, batch_begin + batch_size)]
            }
        )

        test_fn = theano.function(
            inputs=[index],
            outputs=self.y_pred,
            givens={
                self.input: test_set_x[batch_begin: T.switch(T.eq(index, n_test_batches-1), -1, batch_begin + batch_size)]
            }
        )

        pred_fn = theano.function(
            inputs=[self.input],
            outputs=self.y_pred
        )

        return train_fn, valid_fn, test_fn, pred_fn

    def update_learning_rate(self, rate_mul):
        """ Updates learning rate

        :type rate_multiplier: float
        :param rate_multiplier: rate at which the learning is updated
        """
        new_lr = np.clip(self.learning_rate * rate_mul, 1e-6, np.inf)
        self.learning_rate = new_lr

    def update_momentum(self, rate_mul):
        """ Updates momentum

        :type rate_multiplier: float
        :param rate_multiplier: rate at which the momentum is updated
        """
        new_momentum = np.clip(self.momentum * (1+rate_mul), 0, 0.999)
        self.momentum = new_momentum

class ReLuLayer(Layer):
    """ Rectified Linear Unit layer class """
    def __init__(self, input, n_in, n_out, output=None, rng=None, W=None, b=None, activation=T.nnet.relu):
        """ Initialize parameters of the layer """

        super().__init__(input, output, n_in, n_out)

        if rng is None:
            rng = np.random.RandomState(1234)

        self.rng = rng
        self.activation = activation

        if W is None:
            W_value = glorot(self.rng, self.n_in, self.n_out)
            W = theano.shared(W_value, name='W', borrow=True)

        if b is None:
            b_value = uniform(self.rng, (self.n_out,))
            b = theano.shared(b_value, name='b', borrow=True)

        self.params.extend((W,b))
        self.W = W
        self.b = b

        if activation is None:
            self.p_y_given_x = T.dot(self.input, self.W) + self.b
        else:
            self.p_y_given_x = self.activation(T.dot(self.input, self.W) + self.b)

        self.y_pred = T.round(self.p_y_given_x)

    def mean_squared_error(self, y):
        """ Mean Squared Error of prediction

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example
        """
        return T.mean((self.p_y_given_x-y)**2)

class LogisticLayer(Layer):
    """ Logistic regression layer class """
    def __init__(self, input, n_in, n_out, output=None, rng=None, W=None, b=None, activation=T.nnet.softmax):
        """ Initialize parameters of the layer """

        super().__init__(input, output, n_in, n_out)

        if rng is None:
            rng = np.random.RandomState(1234)

        self.rng = rng
        self.activation = activation

        if W is None:
            W_value = glorot(self.rng, self.n_in, self.n_out)
            W = theano.shared(W_value, name='W', borrow=True)

        if b is None:
            b_value = uniform(self.rng, (self.n_out,))
            b = theano.shared(b_value, name='b', borrow=True)

        self.params.extend((W,b))
        self.W = W
        self.b = b

        if activation is None:
            self.p_y_given_x = T.dot(self.input, self.W) + self.b
        else:
            self.p_y_given_x = self.activation(T.dot(self.input, self.W) + self.b)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

    def negative_log_likelihood(self, y):
        """ Negative log-likelihood of the prediction

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example
        """
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """zero one loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example
        """
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type))

        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class HiddenLayer(Layer):
    """ Hidden layer class """
    def __init__(self, input, n_in, n_out, output=None, rng=None, W=None, b=None, activation=None):
        """ Initialize parameters of the layer """
        super().__init__(input, output, n_in, n_out)

        if rng is None:
            rng = np.random.RandomState(1234)

        self.rng = rng
        self.activation = activation

        if W is None:
            W_value = glorot(self.rng, self.n_in, self.n_out)
            W = theano.shared(W_value, name='W', borrow=True)

        if b is None:
            b_value = uniform(self.rng, (self.n_out,))
            b = theano.shared(b_value, name='b', borrow=True)

        self.params.extend((W, b))
        self.W = W
        self.b = b

        if self.activation is None:
            self.output = T.dot(input, self.W) + self.b
        else:
            self.output = self.activation(T.dot(input, self.W) + self.b)

class RecurrentLayer(Layer):
    """ Recurrent Layer class """
    def __init__(self, input, n_in, n_out, output=None, rng=None, W=None, b=None, H=None, h0=None, activation=T.nnet.sigmoid):
        """ Initialize parameters of the layer """
        super().__init__(input, output, n_in, n_out)

        if rng is None:
            rng = np.random.RandomState(1234)

        self.rng = rng
        self.activation = activation

        if W is None:
            W_value = glorot(self.rng, self.n_in, self.n_out)
            W = theano.shared(W_value, name='W', borrow=True)

        if b is None:
            b_value = uniform(self.rng, (self.n_out,))
            b = theano.shared(b_value, name='b', borrow=True)

        self.params.extend((W, b))
        self.W = W
        self.b = b

        if H is None:
            H_value = glorot(self.rng, self.n_out, self.n_out)
            H = theano.shared(H_value, name='H', borrow=True)

        if h0 is None:
            h0_value = uniform(self.rng, (self.n_out,))
            h0 = theano.shared(h0_value, name='h0', borrow=True)

        self.params.extend((H, h0))
        self.H = H
        self.h0 = h0

        # If we used batches, we have to permute the first and second dimension.
        self.output, _ = theano.scan(self.step,
            sequences=self.input.dimshuffle(1, 0, 2),
            outputs_info=T.alloc(self.h0, self.input.shape[0], self.n_out)
        )

    def step(self, x_t, h_tm1):
        h_t = self.activation(T.dot(x_t, self.W) + T.dot(h_tm1, self.H) + self.b)
        return h_t

class LSTMCell(Layer):
    """ LSTM cell """
    def __init__(self, input, n_in, n_cell, n_out, output=None, rng=None, activation=T.nnet.sigmoid, W_i=None, W_f=None, W_c=None, W_o=None, b_i=None, b_f=None, b_c=None, b_o=None, H_i=None, H_f=None, H_c=None, H_o=None, C_i=None, C_f=None, C_o=None, c0=None, h0=None):
        """ Initialize parameters of the layer """
        super().__init__(input, output, n_in, n_out)

        if rng is None:
            rng = np.random.RandomState(1234)

        self.rng = rng
        self.activation = activation
        self.n_cell = n_cell

        if W_i is None:
            W_value = glorot(self.rng, self.n_in, self.n_cell)
            W_i = theano.shared(W_value, name='W_i', borrow=True)

        if W_f is None:
            W_value = glorot(self.rng, self.n_in, self.n_cell)
            W_f = theano.shared(W_value, name='W_f', borrow=True)

        if W_c is None:
            W_value = glorot(self.rng, self.n_in, self.n_cell)
            W_c = theano.shared(W_value, name='W_c', borrow=True)

        if W_o is None:
            W_value = glorot(self.rng, self.n_in, self.n_out)
            W_o = theano.shared(W_value, name='W_o', borrow=True)

        if b_i is None:
            b_value = uniform(self.rng, (self.n_cell,))
            b_i = theano.shared(b_value, name='b_i', borrow=True)

        if b_f is None:
            b_value = uniform(self.rng, (self.n_cell,), 0.5, 1.5)
            b_f = theano.shared(b_value, name='b_f', borrow=True)

        if b_c is None:
            b_value = uniform(self.rng, (self.n_cell,))
            b_c = theano.shared(b_value, name='b_c', borrow=True)

        if b_o is None:
            b_value = uniform(self.rng, (self.n_out,))
            b_o = theano.shared(b_value, name='b_o', borrow=True)

        self.params.extend((W_i, W_f, W_c, W_o, b_i, b_f, b_c, b_o))
        self.W_i = W_i
        self.W_f = W_f
        self.W_c = W_c
        self.W_o = W_o
        self.b_i = b_i
        self.b_f = b_f
        self.b_c = b_c
        self.b_o = b_o

        if H_i is None:
            H_value = glorot(self.rng, self.n_out, self.n_cell)
            H_i = theano.shared(H_value, name='H_i', borrow=True)

        if H_f is None:
            H_value = glorot(self.rng, self.n_out, self.n_cell)
            H_f = theano.shared(H_value, name='H_f', borrow=True)

        if H_c is None:
            H_value = glorot(self.rng, self.n_out, self.n_cell)
            H_c = theano.shared(H_value, name='H_c', borrow=True)

        if H_o is None:
            H_value = glorot(self.rng, self.n_out, self.n_out)
            H_o = theano.shared(H_value, name='H_o', borrow=True)

        self.params.extend((H_i, H_f, H_c, H_o))
        self.H_i = H_i
        self.H_f = H_f
        self.H_c = H_c
        self.H_o = H_o

        if C_i is None:
            C_value = glorot(self.rng, self.n_cell, self.n_cell)
            C_i = theano.shared(C_value, name='C_i', borrow=True)

        if C_f is None:
            C_value = glorot(self.rng, self.n_cell, self.n_cell)
            C_f = theano.shared(C_value, name='C_f', borrow=True)

        if C_o is None:
            C_value = glorot(self.rng, self.n_cell, self.n_out)
            C_o = theano.shared(C_value, name='C_o', borrow=True)

        self.params.extend((C_i, C_f, C_o))
        self.C_i = C_i
        self.C_f = C_f
        self.C_o = C_o

        if c0 is None:
            c0_value = uniform(self.rng, (self.n_cell,))
            c0 = theano.shared(c0_value, name='c0', borrow=True)

        if h0 is None:
            h0_value = uniform(self.rng, (self.n_out,))
            h0 = theano.shared(h0_value, name='h0', borrow=True)

        self.params.extend((c0, h0))
        self.c0 = c0
        self.h0 = h0

        # If we used batches, we have to permute the first and second dimension.
        [self.cell, self.output], _ = theano.scan(self.step,
            sequences=self.input.dimshuffle(1, 0, 2),
            outputs_info=[T.alloc(self.c0, self.input.shape[0], self.n_cell),
            T.alloc(self.h0, self.input.shape[0], self.n_out)]
        )

    def step(self, x_t, c_tm1, h_tm1):
        i_t = T.nnet.sigmoid(T.dot(x_t, self.W_i) + T.dot(h_tm1, self.H_i) + \
              T.dot(c_tm1, self.C_i) + self.b_i)
        f_t = T.nnet.sigmoid(T.dot(x_t, self.W_f) + T.dot(h_tm1, self.H_f) + \
              T.dot(c_tm1, self.C_f) + self.b_f)
        c_t = f_t * c_tm1 + i_t * self.activation(T.dot(x_t, self.W_c) + \
              T.dot(h_tm1, self.H_c) + self.b_c)
        o_t = T.nnet.sigmoid(T.dot(x_t, self.W_o) + T.dot(h_tm1, self.H_o) + \
              T.dot(c_t, self.C_o) + self.b_o)
        h_t = o_t * self.activation(o_t)
        return c_t, h_t

class LSTM(Network):
    """ LSTM neural network class """
    def __init__(self, input, output, n_in, n_cell, n_hidden, n_out, drop, rng=None, optimizer=sgd, lstm_activation=T.tanh, output_activation=T.nnet.relu):
        """ Initialize parameters of the recurrent layer """
        super().__init__(input, output, n_in, n_out, rng, optimizer)

        if rng is None:
            rng = np.random.RandomState(1234)

        if not isinstance(n_hidden, list):
            raise TypeError('n_hidden should be a list of ints')

        self.rng = rng
        self.n_hidden = n_hidden
        self.n_cell = n_cell
        self.optimizer = optimizer
        self.lstm_activation = lstm_activation
        self.output_activation = output_activation

        self.n_layers = len(self.n_hidden)
        self.lstm_layers = []
        self.lstm_drop_layers = []

        for i in range(self.n_layers):
            if i == 0:
                input_size = self.n_in
                layer_input = self.input
                drop_layer_input = self.input
            else:
                input_size = self.n_hidden[i - 1]
                layer_input = self.lstm_layers[-1].output
                layer_input = layer_input.dimshuffle(1, 0, 2)
                drop_layer_input = self.lstm_drop_layers[-1].output
                drop_layer_input = drop_layer_input.dimshuffle(1, 0, 2)

            lstm_layer = LSTMCell(
                input=layer_input,
                n_in=input_size,
                n_cell=self.n_cell,
                n_out=self.n_hidden[i],
                rng=self.rng,
                activation=self.lstm_activation
            )
            self.lstm_layers.append(lstm_layer)
            self.params.extend(lstm_layer.params)

            lstm_drop_layer = LSTMCell(
                input=dropout(self.rng, drop_layer_input, p=drop[i]),
                n_in=input_size,
                n_cell=self.n_cell,
                n_out=self.n_hidden[i],
                rng=self.rng,
                W_i=lstm_layer.W_i, W_f=lstm_layer.W_f,
                W_c=lstm_layer.W_c, W_o=lstm_layer.W_o,
                b_i=lstm_layer.b_i, b_f=lstm_layer.b_f,
                b_c=lstm_layer.b_c, b_o=lstm_layer.b_o,
                H_i=lstm_layer.H_i, H_f=lstm_layer.H_f,
                H_c=lstm_layer.H_c, H_o=lstm_layer.H_o,
                C_i=lstm_layer.C_i, C_f=lstm_layer.C_f,
                C_o=lstm_layer.C_o, c0=lstm_layer.c0,
                h0=lstm_layer.h0,
                activation=self.lstm_activation
            )
            self.lstm_drop_layers.append(lstm_drop_layer)

        self.output_layer = ReLuLayer(
            input=self.lstm_layers[-1].output,
            n_in=self.n_hidden[-1],
            n_out=self.n_out,
            rng=self.rng,
            activation=self.output_activation
        )

        self.output_drop_layer = ReLuLayer(
            input=dropout(self.rng, self.lstm_drop_layers[-1].output, p=drop[-1]),
            n_in=self.n_hidden[-1],
            n_out=self.n_out,
            rng=self.rng,
            W=self.output_layer.W,
            b=self.output_layer.b,
            activation=self.output_activation
        )

        self.params.extend(self.output_layer.params)

        self.cost = self.mean_squared_error(self.output)
        self.errors = self.mean_errors(self.output)
        self.y_pred = self.output_layer.y_pred

    def mean_squared_error(self, y):
        """ Mean Squared Error of prediction

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example
        """
        y = y.dimshuffle(1, 0, 2)[-1:]
        return T.mean((self.output_drop_layer.p_y_given_x[-1:]-y)**2)

    def mean_errors(self, y):
        """ Mean Squared Error of prediction

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example
        """
        y = y.dimshuffle(1, 0, 2)[-1:]
        return T.mean((self.output_layer.p_y_given_x[-1:]-y)**2)

class RNN(Network):
    """ Recurrent neural network class """
    def __init__(self, input, output, n_in, n_hidden, n_out, drop, rng=None, optimizer=sgd, recurrent_activation=T.tanh, output_activation=T.nnet.relu):
        """ Initialize parameters of the recurrent layer """
        super().__init__(input, output, n_in, n_out, rng, optimizer)

        if rng is None:
            rng = np.random.RandomState(1234)

        if not isinstance(n_hidden, list):
            raise TypeError('n_hidden should be a list of ints')

        self.rng = rng
        self.n_hidden = n_hidden
        self.optimizer = optimizer
        self.recurrent_activation = recurrent_activation
        self.output_activation = output_activation

        self.n_layers = len(self.n_hidden)
        self.recurrent_layers = []
        self.recurrent_dropout_layers = []

        for i in range(self.n_layers):
            if i == 0:
                input_size = self.n_in
                layer_input = self.input
            else:
                input_size = self.n_hidden[i - 1]
                layer_input = self.recurrent_layers[-1].output
                layer_input = layer_input.dimshuffle(1, 0, 2)

            recurrent_layer = RecurrentLayer(
                input=layer_input,
                n_in=input_size,
                n_out=self.n_hidden[i],
                rng=self.rng,
                activation=self.recurrent_activation
            )
            self.recurrent_layers.append(recurrent_layer)
            self.params.extend(recurrent_layer.params)

            recurrent_dropout_layer = RecurrentLayer(
                input=dropout(self.rng, recurrent_layer.input, p=drop[i]),
                n_in=input_size,
                n_out=self.n_hidden[i],
                W=recurrent_layer.W,
                b=recurrent_layer.b,
                H=recurrent_layer.H,
                h0=recurrent_layer.h0,
                rng=self.rng,
                activation=recurrent_layer.activation
            )
            self.recurrent_dropout_layers.append(recurrent_dropout_layer)

        self.output_layer = ReLuLayer(
            input=self.recurrent_layers[-1].output,
            n_in=self.n_hidden[-1],
            n_out=self.n_out,
            rng=self.rng,
            activation=self.output_activation
        )

        self.params.extend(self.output_layer.params)

        self.output_dropout_layer = ReLuLayer(
            input=self.recurrent_dropout_layers[-1].output,
            n_in=self.n_hidden[-1],
            n_out=self.n_out,
            rng=self.rng,
            W=self.output_layer.W,
            b=self.output_layer.b,
            activation=self.output_layer.activation
        )

        self.cost = self.mean_squared_error(self.output)
        self.errors = self.mean_errors(self.output)
        self.y_pred = self.output_layer.y_pred

    def mean_squared_error(self, y):
        """ Mean Squared Error of prediction

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example
        """
        y = y.dimshuffle(1, 0, 2)[-1:]
        return T.mean((self.output_dropout_layer.p_y_given_x[-1:]-y)**2)

    def mean_errors(self, y):
        """ Mean Squared Error of prediction

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example
        """
        y = y.dimshuffle(1, 0, 2)[-1:]
        return T.mean((self.output_layer.p_y_given_x[-1:]-y)**2)

class MLP(Network):
    """ Multi layer perceptron class """
    def __init__(self, n_in, n_hidden, n_out, input=None, output=None, rng=None, optimizer=sgd_nesterov, activation=T.tanh, out_function="softmax"):
        """ Initialize the parameters of the MLP """
        super().__init__(input, output, n_in, n_out, rng, optimizer)

        if self.input is None:
            self.input = T.matrix('x')

        if self.output is None:
            self.output = T.ivector('y')

        if rng is None:
            rng = np.random.RandomState(1234)

        self.n_hidden = n_hidden
        self.n_layers = len(self.n_hidden)
        self.hidden_layers = []
        self.rng = rng
        self.optimizer = optimizer
        self.activation = activation

        if not isinstance(n_hidden, list):
            raise TypeError('n_hidden should be a list of ints')

        for i in range(self.n_layers):
            if i == 0:
                input_size = self.n_in
                layer_input = self.input
            else:
                input_size = self.n_hidden[i - 1]
                layer_input = self.hidden_layers[-1].output

            hidden_layer = HiddenLayer(
                input=layer_input,
                n_in=input_size,
                n_out=self.n_hidden[i],
                rng=self.rng,
                activation=self.activation
            )

            self.hidden_layers.append(hidden_layer)
            self.params.extend(hidden_layer.params)

        if out_function is "softmax":
            self.out_layer = LogisticLayer(
                input=self.hidden_layers[-1].output,
                n_in=self.n_hidden[-1],
                n_out=self.n_out,
                rng=self.rng
            )
            self.cost = self.out_layer.negative_log_likelihood(self.output)
            self.errors = self.out_layer.errors(self.output)

        elif out_function is "relu":
            self.out_layer = ReLuLayer(
                input=self.hidden_layers[-1].output,
                n_in=self.n_hidden[-1],
                n_out=self.n_out,
                rng=self.rng,
                activation=T.nnet.relu
            )
            self.cost = self.out_layer.mean_squared_error(self.output)
            self.errors = self.out_layer.mean_squared_error(self.output)

        self.params.extend(self.out_layer.params)
        self.p_y_given_x = self.out_layer.p_y_given_x
        self.y_pred = self.out_layer.y_pred

class DropoutMLP(MLP):
    """ Multi layer perceptron class with dropout """
    def __init__(self, n_in, n_hidden, n_out, drop, input=None, output=None, rng=None, optimizer=sgd_nesterov, activation=T.tanh, out_function="softmax"):
        """ Initialize the parameters of the MLP

        :type dropout: list of floats
        :param dropout: dropout probabilities, must contain at least 1 value
        """

        super().__init__(n_in, n_hidden, n_out, input, output, rng, optimizer, activation, out_function)

        self.dropout_layers = []

        if not isinstance(drop, list):
            raise TypeError('dropout should be a list of floats')

        for i, hidden_layer in enumerate(self.hidden_layers):
            if i == 0:
                layer_input = self.input
            else:
                layer_input = self.dropout_layers[-1].output

            dropout_layer = HiddenLayer(
                input=dropout(self.rng, layer_input, p=drop[i]),
                n_in=hidden_layer.n_in,
                n_out=hidden_layer.n_out,
                activation=hidden_layer.activation,
                W=hidden_layer.W,
                b=hidden_layer.b
            )
            self.dropout_layers.append(dropout_layer)

        if out_function is "softmax":
            self.dropout_out_layer = LogisticLayer(
                input=dropout(self.rng, self.dropout_layers[-1].output,
                    p=drop[-1]),
                n_in=self.out_layer.n_in,
                n_out=self.out_layer.n_out,
                W=self.out_layer.W,
                b=self.out_layer.b
            )
            self.cost = self.dropout_out_layer.negative_log_likelihood(self.output)
        elif out_function is "relu":
            self.dropout_out_layer = ReLuLayer(
                input=dropout(self.rng, self.dropout_layers[-1].output,
                    p=drop[-1]),
                n_in=self.out_layer.n_in,
                n_out=self.out_layer.n_out,
                W=self.out_layer.W,
                b=self.out_layer.b,
                activation=T.nnet.relu
            )
            self.cost = self.dropout_out_layer.mean_squared_error(self.output)

class RBM(object):
    """ Generic Restricted Boltzmann Machine (RBM) container class """
    def __init__(self, input, n_visible, n_hidden):
        """ Initialize the parameters of the RBM

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input

        :type output: theano.tensor.TensorType
        :param output: symbolic variable that describes the output

        :type n_visible: list of ints
        :param n_visible: number of visible units

        :type n_hidden: list of ints
        :param n_hidden: number of hidden units
        """

        # keeping track of the model
        self.input = input
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.params = []

class RBMLayer(Network, RBM):
    """ RBM layer class """
    def __init__(self, input, n_visible, n_hidden, output=None, rng=None, theano_rng=None, W=None, hbias=None, vbias=None, optimizer=sgd):
        """ Initialize the parameters of the RBM Layer """

        Network.__init__(self, input, output, n_visible, n_hidden, rng, optimizer)
        RBM.__init__(self, input, n_visible, n_hidden)

        if theano_rng is None:
            theano_rng = RandomStreams(self.rng.randint(2 ** 30))

        self.theano_rng = theano_rng

        if W is None:
            W_value = glorot(rng, self.n_visible, self.n_hidden)
            W = theano.shared(W_value, name='W', borrow=True)

        if hbias is None:
            hbias_value = uniform(self.rng, (self.n_hidden))
            hbias = theano.shared(hbias_value, name='hbias', borrow=True)

        if vbias is None:
            vbias_value = uniform(self.rng, (self.n_visible))
            vbias = theano.shared(vbias_value, name='vbias', borrow=True)

        self.params.extend((W, hbias, vbias))
        self.W = W
        self.hbias = hbias
        self.vbias = vbias

    def free_energy(self, v_sample):
        """ Function to compute binary-binary free energy

        :type v_sample: theano.tensor.TensorType
        :param v_sample: symbolic variable that describes the sample input
        """

        wx_b = self.hbias + T.dot(v_sample, self.W)
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.nnet.softplus(wx_b) , axis=1)

        return - hidden_term - vbias_term

    def propup(self, vis):
        """ This function propagates the visible units activation upwards to
        the hidden units """

        pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias

        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        """ This function infers state of hidden units given visible units """

        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)

        h1_sample = self.theano_rng.binomial(
            size=h1_mean.shape, n=1, p=h1_mean, dtype=theano.config.floatX
        )

        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        """ This function propagates the hidden units activation downwards to
        the visible units """

        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias

        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        """ This function infers state of visible units given hidden units """

        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)

        v1_sample = self.theano_rng.binomial(
            size=v1_mean.shape, n=1, p=v1_mean, dtype=theano.config.floatX
        )

        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        """ This function implements one step of Gibbs sampling,
            starting from the hidden state"""
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        """ This function implements one step of Gibbs sampling,
            starting from the visible state"""
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def get_cost_updates(self, learning_rate, momentum, persistent=None, k=1):
        """This functions implements one step of CD-k or PCD-k

        :param lr: learning rate used to train the RBM

        :param persistent: None for CD. For PCD, shared variable
            containing old state of Gibbs chain. This must be a shared
            variable of size (batch size, number of hidden units).

        :param k: number of Gibbs steps to do in CD-k/PCD-k
        """

        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)

        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent

        ([pre_sigmoid_nvs, nv_means, nv_samples,
          pre_sigmoid_nhs, nh_means, nh_samples],
         updates
        ) = theano.scan(
            self.gibbs_hvh,
            outputs_info=[None, None, None, None, None, chain_start],
            n_steps=k,
            name="gibbs_hvh"
        )

        chain_end = nv_samples[-1]
        self.out_test = pre_sigmoid_nvs
        fe_x = T.mean(self.free_energy(self.input)) - T.mean(
            self.free_energy(chain_end))

        gparams = T.grad(fe_x, self.params, consider_constant=[chain_end])
        opt = self.optimizer(self.params)
        opt_update = opt.updates(self.params, gparams, learning_rate, momentum)

        if persistent:
            updates[persistent] = nh_samples[-1]
            print('... persistent CD-%d training' % k)
            monitoring_cost = self.get_pseudo_log_likelihood(updates)
        else:
            print('... CD-%d training' % k)
            monitoring_cost = self.get_cross_entropy_cost(pre_sigmoid_nvs[-1])

        for val in updates:
            opt_update.append((val, updates[val]))

        return monitoring_cost, opt_update

    def get_pseudo_log_likelihood(self, updates):
        """ Stochastic approximation to the pseudo-likelihood """

        bit_i_idx = theano.shared(value=0, name='bit_i_idx')
        xi = T.round(self.input)
        fe_xi = self.free_energy(xi)
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])
        fe_xi_flip = self.free_energy(xi_flip)

        cost =  T.mean(
            T.log(T.nnet.sigmoid(fe_xi_flip-fe_xi)) * self.n_visible
        )
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

        return cost

    def get_cross_entropy_cost(self, pre_sigmoid_nv):
        """Approximation to the reconstruction error """

        return T.mean(
            T.sum(
                self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
                (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
                axis=1
            )
        )

    def build_fns(self, dataset, batch_size, learning_rate, momentum,
    persistent=True, k=1):
        """Generates theano functions 'train_fn', 'valid_fn' and 'test_fn'

        :type dataset: list of pairs of theano.tensor.TensorType
        :param dataset: It is a list that contain all the datasets

        :type batch_size: int
        :param batch_size: size of a minibatch

        :type learning_rate: float
        :param learning_rate: learning rate used during training

        :type momentum: float
        :param momentum: momentum used during training
        """

        train_set_x, train_set_y = dataset[0]
        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

        print('... building the model')
        index = T.lscalar('index') # index to a [mini]batch

        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size

        self.learning_rate = learning_rate
        self.momentum = momentum

        if persistent:
            self.persistent_chain = theano.shared(
                zeros((batch_size, self.n_hidden)), name='persistent chain', borrow=True)
        else:
            self.persistent_chain = None

        cost, updates = self.get_cost_updates(
            learning_rate=self.learning_rate, momentum=self.momentum,
            persistent=self.persistent_chain, k=k)

        train_fn = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                self.input: train_set_x[batch_begin: batch_end]
            }
        )

        return train_fn

class GBRBMLayer(RBMLayer):
    """ Gaussian-Bernoulli RBM layer class """
    def __init__(self, input, n_visible, n_hidden, output=None, rng=None, theano_rng=None, W=None, hbias=None, vbias=None, optimizer=sgd):
        """ Initialize the parameters of the GBRBM Layer """

        super().__init__(input, n_visible, n_hidden, output, rng, theano_rng, W, hbias, vbias, optimizer)

    def free_energy(self, v_sample):
        """ Function to compute gaussian-binary free energy

        :type v_sample: theano.tensor.TensorType
        :param v_sample: symbolic variable that describes the sample input
        """

        wx_b = self.hbias + T.dot(v_sample, self.W)
        hidden_term = T.sum(T.nnet.softplus(wx_b) , axis=1)
        visible_term = T.sum(0.5 * T.sqr(v_sample - self.vbias), axis=1)

        return visible_term - hidden_term

    def propdown(self, hid):
        """ This function propagates the hidden units activation downwards to
        the visible units """

        mean_activation = T.dot(hid, self.W.T) + self.vbias

        return mean_activation

    def sample_v_given_h(self, h0_sample):
        """ This function infers state of visible units given hidden units """

        v1_mean = self.propdown(h0_sample)
        v1_sample = v1_mean

        return [v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        """ This function implements one step of Gibbs sampling,
            starting from the hidden state"""
        v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        """ This function implements one step of Gibbs sampling,
            starting from the visible state"""
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                v1_mean, v1_sample]

    def get_cost_updates(self, learning_rate, momentum, persistent=None, k=1):
        """This functions implements one step of CD-k or PCD-k

        :param lr: learning rate used to train the RBM

        :param persistent: None for CD. For PCD, shared variable
            containing old state of Gibbs chain. This must be a shared
            variable of size (batch size, number of hidden units).

        :param k: number of Gibbs steps to do in CD-k/PCD-k
        """

        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)

        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent

        ([nv_means, nv_samples,
          pre_sigmoid_nhs, nh_means, nh_samples],
         updates
        ) = theano.scan(
            self.gibbs_hvh,
            outputs_info=[None, None, None, None, chain_start],
            n_steps=k,
            name="gibbs_hvh"
        )

        chain_end = nv_samples[-1]

        fe_x = T.mean(self.free_energy(self.input)) - T.mean(
            self.free_energy(chain_end))

        gparams = T.grad(fe_x, self.params, consider_constant=[chain_end])
        opt = self.optimizer(self.params)
        opt_update = opt.updates(self.params, gparams, learning_rate, momentum)

        if persistent:
            updates[persistent] = nh_samples[-1]
            print('... persistent CD-%d training' % k)
            monitoring_cost = self.get_pseudo_log_likelihood(updates)
        else:
            print('... CD-%d training' % k)
            monitoring_cost = self.get_mean_cost(nv_means[-1])

        for val in updates:
            opt_update.append((val, updates[val]))

        return monitoring_cost, opt_update

    def get_mean_cost(self, pre_sigmoid_nv):
        """Approximation to the reconstruction error """

        return T.mean(T.sum(T.sqr(self.input - pre_sigmoid_nv), axis=1))

class DBN(DropoutMLP):
    def __init__(self, n_visible, n_hidden, n_out, drop, input=None, output=None, rng=None, theano_rng=None, optimizer=sgd, activation=T.nnet.sigmoid, out_function="softmax", gaussian_rbm=False):
        """ Initialize the parameters of the DBN Layer """

        DropoutMLP.__init__(self, n_visible, n_hidden, n_out, drop, input, output, rng, optimizer, activation, out_function)

        if theano_rng is None:
            theano_rng = RandomStreams(self.rng.randint(2 ** 30))

        self.theano_rng=theano_rng
        self.rbm_layers = []

        if not isinstance(n_hidden, list):
            raise TypeError('n_hidden should be a list of ints')

        for hidden_layer in self.hidden_layers:
            if gaussian_rbm is True:
                rbm_layer = GBRBMLayer(
                    input=hidden_layer.input,
                    n_visible=hidden_layer.n_in,
                    n_hidden=hidden_layer.n_out,
                    rng=self.rng,
                    W=hidden_layer.W,
                    hbias=hidden_layer.b,
                    optimizer=self.optimizer
                )
            else:
                layer_input = self.theano_rng.binomial(
                    size=hidden_layer.input.shape, n=1, p=hidden_layer.input, dtype=theano.config.floatX
                )
                rbm_layer = RBMLayer(
                    input=layer_input,
                    n_visible=hidden_layer.n_in,
                    n_hidden=hidden_layer.n_out,
                    rng=self.rng,
                    W=hidden_layer.W,
                    hbias=hidden_layer.b,
                    optimizer=self.optimizer
                )
            self.rbm_layers.append(rbm_layer)

    def build_rbm_fns(self, dataset, batch_size, learning_rate, momentum, persistent=True, k=1):
        """Generates theano functions 'train_fn' """

        train_set_x, train_set_y = dataset[0]
        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

        print('... building the model')
        index = T.lscalar('index') # index to a [mini]batch

        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size

        self.learning_rate = learning_rate
        self.momentum = momentum

        rbm_fns = []
        for i, rbm_layer in enumerate(self.rbm_layers):

            if persistent:
                self.persistent_chain = theano.shared(
                    zeros((batch_size, self.n_hidden[i])),
                    name='persistent chain', borrow=True)
            else:
                self.persistent_chain = None

            cost, updates = rbm_layer.get_cost_updates(
                learning_rate=self.learning_rate, momentum=self.momentum,
                persistent=self.persistent_chain, k=k)

            train_fn = theano.function(
                inputs=[index],
                outputs=cost,
                updates=updates,
                givens={
                    self.input: train_set_x[batch_begin: batch_end]
                }
            )
            rbm_fns.append(train_fn)

        return rbm_fns

def load_data(dataset, with_h5f=True, cast_y_to_int32=True):
    ''' Loads the dataset
    :type dataset: string
    :param dataset: the path to the dataset
    '''
    print('... loading data')
    if with_h5f:
        with h5py.File(dataset,'r') as h5f:
            train_set = h5f['train_set_x'][:], h5f['train_set_y'][:]
            valid_set = h5f['valid_set_x'][:], h5f['valid_set_y'][:]
            test_set = h5f['test_set_x'][:], h5f['test_set_y'][:]
    else:
        with gzip.open(dataset, 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables """
        data_x, data_y = data_xy
        shared_x = theano.shared(
            np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(
            np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        if cast_y_to_int32:
            return shared_x, T.cast(shared_y, 'int32')
        else:
            return shared_x, shared_y

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]

def test_dbn(datafile='mnist.pkl.gz', pretrain_epochs=50, n_epochs=50,
             batch_size=32, n_visible=784, n_hidden=[32], n_out=10,
             dropout=[0.0], k=3, persistent=True, gaussian_rbm=False,
             out_function="softmax", pretrain_learning_rate=0.1,
             learning_rate=0.03, momentum=0.5, optimizer=sgd, lr_decay=1.,
             ramp=0., model_file='best_model.pkl', improvement_threshold=0.995,
             with_h5f=False, cast_y_to_int32=True):

    dataset = load_data(datafile, with_h5f, cast_y_to_int32)
    train_set_x, train_set_y = dataset[0]
    valid_set_x, valid_set_y = dataset[1]
    test_set_x, test_set_y = dataset[2]
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    x = T.matrix('x')
    y = T.matrix('y')

    model = DBN(input=x, output=y, n_visible=n_visible,
        n_hidden=n_hidden, n_out=n_out, drop=dropout, optimizer=optimizer,
        activation=T.nnet.sigmoid, out_function=out_function,
        gaussian_rbm=gaussian_rbm)

    ###################
    # PRE-TRAIN MODEL #
    ###################
    start_time = timeit.default_timer()
    if pretrain_epochs > 0:
        print('... getting the pre-training functions')
        pretrain_fns = model.build_rbm_fns(dataset, batch_size,
            learning_rate, momentum, persistent=persistent, k=k)

        print('... pre-training the model')
        # Pre-train layer-wise
        for i in range(model.n_layers):
            model.learning_rate = pretrain_learning_rate
            model.momentum = momentum
            for epoch in range(pretrain_epochs):
                c = []
                for batch_index in range(n_train_batches):
                    c.append(pretrain_fns[i](batch_index))
                print('Pre-training layer %i, epoch %d, cost %f' %
                    (i, epoch, np.mean(c)))
                model.update_learning_rate(lr_decay)
                model.update_momentum(ramp)
                print('learning rate: %f, momentum: %f' %
                    (model.learning_rate, model.momentum))

        end_time = timeit.default_timer()
        print('The pre-training code for file ' + os.path.split(__file__)[1] +
              ' ran for %.2fm' % ((end_time - start_time) / 60.), file=sys.stderr)

    ###############
    # TRAIN MODEL #
    ###############
    print('... getting the training functions')
    train_fn, valid_fn, test_fn = model.build_fns(
        dataset, batch_size, learning_rate, momentum)

    print('... training the model')
    # early stopping
    patience = n_epochs // 2 * n_train_batches
    patience_increase = 2
    validation_frequency = np.min((n_train_batches, patience // 2))

    done_looping = False
    best_validation_loss = np.inf
    test_score = 0.
    epoch = 0
    done_looping = False

    start_time = timeit.default_timer()
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if (iter + 1) % validation_frequency == 0:

                if out_function is "relu":
                    validation_loss = np.sqrt(np.mean(
                        [valid_fn(i) for i in range(n_valid_batches)]))
                    print('epoch %i, minibatch %i/%i, validation error %f' %
                            (epoch, minibatch_index + 1, n_train_batches,
                            validation_loss))

                elif out_function is "softmax":
                    validation_loss = np.mean(
                        [valid_fn(i) for i in range(n_valid_batches)])
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                            (epoch, minibatch_index + 1, n_train_batches,
                            validation_loss * 100.))

                if validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if (validation_loss < best_validation_loss *
                        improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = validation_loss
                    test_pred = np.concatenate([test_fn(i) for i in range(n_test_batches)]).ravel()

                    print('test dataset prediction', test_pred, test_pred.shape)
                    print('learning rate: %f, momentum: %f' %
                        (model.learning_rate, model.momentum))

                    # save the best model
                    with open(model_file, 'wb') as f:
                        pickle.dump(model, f)

            if patience <= iter:
                done_looping = True
                break

        # update learning rate every epoch
        model.update_learning_rate(lr_decay)
        model.update_momentum(ramp)

    end_time = timeit.default_timer()
    if out_function is "relu":
        print('Optimization complete with best validation score of %f, with test performance %f' % (best_validation_loss, test_score))
    elif out_function is "softmax":
        print('Optimization complete with best validation score of %f %%, with test performance %f %%' % (best_validation_loss * 100., test_score * 100.))
    print('The code run for %d epochs, with %f epochs/sec' %
        (epoch, 1. * epoch / (end_time - start_time)))
    print(('The code for file ' + os.path.split(__file__)[1] +
           ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)

def test_mlp(datafile='mnist.pkl.gz', n_epochs=20, batch_size=32,
             n_in=784, n_hidden=[100], n_out=10, dropout=None,
             learning_rate=0.1, momentum=0.5, optimizer=sgd, lr_decay=1.0,
             ramp=0.0, model_file='best_model.pkl',
             improvement_threshold=0.995, with_h5f=False, cast_y_to_int32=True):

    dataset = load_data(datafile, with_h5f, cast_y_to_int32)
    train_set_x, train_set_y = dataset[0]
    valid_set_x, valid_set_y = dataset[1]
    test_set_x, test_set_y = dataset[2]
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    x = T.matrix('x')
    y = T.ivector('y')

    if dropout is None:
        model = MLP(input=x, output=y, n_in=n_in, n_hidden=n_hidden,
                    n_out=n_out, optimizer=sgd)
    else:
        model = DropoutMLP(input=x, output=y, n_in=n_in, n_hidden=n_hidden,
                           n_out=n_out, drop=dropout, optimizer=optimizer)

    train_fn, valid_fn, test_fn = model.build_fns(
        dataset, batch_size, learning_rate, momentum)

    ###############
    # TRAIN MODEL #
    ###############
    print('... training the model')
    # early stopping
    patience = n_epochs // 2 * n_train_batches
    patience_increase = 2
    validation_frequency = np.min((n_train_batches, patience // 2))

    done_looping = False
    best_validation_loss = np.inf
    test_score = 0.
    epoch = 0
    done_looping = False

    start_time = timeit.default_timer()
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if (iter + 1) % validation_frequency == 0:
                validation_loss = np.mean(
                    [valid_fn(i) for i in range(n_valid_batches)])

                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                        (epoch, minibatch_index + 1, n_train_batches,
                        validation_loss * 100.))

                if validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if (validation_loss < best_validation_loss *
                        improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = validation_loss
                    test_score = np.mean(
                        [test_fn(i) for i in range(n_test_batches)])

                    print('     epoch %i, minibatch %i/%i, '
                          'test error of best model %f %% ' %
                          (epoch, minibatch_index + 1, n_train_batches,
                          test_score * 100.))

                    # save the best model
                    with open(model_file, 'wb') as f:
                        pickle.dump(model, f)

            if patience <= iter:
                done_looping = True
                break

        # update learning rate every epoch
        model.update_learning_rate(lr_decay)
        model.update_momentum(ramp)
        print('learning rate: %f, momentum: %f' %
            (model.learning_rate, model.momentum))

    end_time = timeit.default_timer()
    print('Optimization complete with best validation score of %f %%, with test performance %f %%' % (best_validation_loss * 100., test_score * 100.))
    print('The code run for %d epochs, with %f epochs/sec' %
        (epoch, 1. * epoch / (end_time - start_time)))
    print(('The code for file ' + os.path.split(__file__)[1] +
           ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)

def test_rnn(datafile='mnist.pkl.gz', n_epochs=20, batch_size=32,
             n_in=784, n_hidden=[100], n_out=10, dropout=None,
             learning_rate=0.1, momentum=0.5, optimizer=sgd, lr_decay=1.0,
             ramp=0.0, model_file='best_model.pkl',
             improvement_threshold=0.995, with_h5f=False, cast_y_to_int32=True):

    dataset = load_data(datafile, with_h5f, cast_y_to_int32)
    train_set_x, train_set_y = dataset[0]
    valid_set_x, valid_set_y = dataset[1]
    test_set_x, test_set_y = dataset[2]
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    x = T.tensor3('x')
    y = T.tensor3('y')
    model = RNN(input=x, output=y, n_in=n_in, n_hidden=n_hidden,
                n_out=n_out, drop=dropout, optimizer=optimizer)

    print('... printing graph')
    theano.printing.pydotprint(model.y_pred, outfile="pydotprint_y.png", var_with_name_simple=True)

    train_fn, valid_fn, test_fn = model.build_fns(
        dataset, batch_size, learning_rate, momentum)



    ###############
    # TRAIN MODEL #
    ###############
    print('... training the model')
    # early stopping
    patience = n_epochs // 2 * n_train_batches
    patience_increase = 2
    validation_frequency = np.min((n_train_batches, patience // 2))

    done_looping = False
    best_validation_loss = np.inf
    test_score = 0.
    epoch = 0
    done_looping = False

    start_time = timeit.default_timer()
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if (iter + 1) % validation_frequency == 0:
                validation_loss = np.sqrt(np.mean(
                    [valid_fn(i) for i in range(n_valid_batches)]))

                print('epoch %i, minibatch %i/%i, validation error %f' %
                        (epoch, minibatch_index + 1, n_train_batches,
                        validation_loss))

                if validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if (validation_loss < best_validation_loss *
                        improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = validation_loss
                    test_score = np.sqrt(np.mean(
                        [test_fn(i) for i in range(n_test_batches)]))

                    print('     epoch %i, minibatch %i/%i, '
                          'test error of best model %f' %
                          (epoch, minibatch_index + 1, n_train_batches,
                          test_score))

                    # save the best model
                    with open(model_file, 'wb') as f:
                        pickle.dump(model, f)

            if patience <= iter:
                done_looping = True
                break

        # update learning rate every epoch
        model.update_learning_rate(lr_decay)
        model.update_momentum(ramp)
        print('learning rate: %f, momentum: %f' %
            (model.learning_rate, model.momentum))

    end_time = timeit.default_timer()
    print('Optimization complete with best validation score of %f, with test performance %f' % (best_validation_loss, test_score))
    print('The code run for %d epochs, with %f epochs/sec' %
        (epoch, 1. * epoch / (end_time - start_time)))
    print(('The code for file ' + os.path.split(__file__)[1] +
           ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)


def test_lstm(datafile='mnist.pkl.gz', n_epochs=20, batch_size=32,
             n_in=784, n_cell=100, n_hidden=[100], n_out=10, dropout=None,
             learning_rate=0.1, momentum=0.5, optimizer=sgd, lr_decay=1.0,
             ramp=0.0, model_file='best_model.pkl',
             improvement_threshold=0.995, with_h5f=False, cast_y_to_int32=True):

    dataset = load_data(datafile, with_h5f, cast_y_to_int32)
    train_set_x, train_set_y = dataset[0]
    valid_set_x, valid_set_y = dataset[1]
    test_set_x, test_set_y = dataset[2]
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    x = T.tensor3('x')
    y = T.tensor3('y')
    model = LSTM(input=x, output=y, n_in=n_in, n_cell=n_cell, n_hidden=n_hidden,
                n_out=n_out, drop=dropout, optimizer=optimizer)

    # print('... printing graph')
    # theano.printing.pydotprint(model.y_pred, outfile="pydotprint_y.png", var_with_name_simple=True)

    train_fn, valid_fn, test_fn = model.build_fns(
        dataset, batch_size, learning_rate, momentum)



    ###############
    # TRAIN MODEL #
    ###############
    print('... training the model')
    # early stopping
    patience = n_epochs // 2 * n_train_batches
    patience_increase = 2
    validation_frequency = np.min((n_train_batches, patience // 2))

    done_looping = False
    best_validation_loss = np.inf
    test_score = 0.
    epoch = 0
    done_looping = False

    start_time = timeit.default_timer()
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if (iter + 1) % validation_frequency == 0:
                validation_loss = np.sqrt(np.mean(
                    [valid_fn(i) for i in range(n_valid_batches)]))

                print('epoch %i, minibatch %i/%i, validation error %f' %
                        (epoch, minibatch_index + 1, n_train_batches,
                        validation_loss))

                if validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if (validation_loss < best_validation_loss *
                        improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = validation_loss
                    test_score = np.sqrt(np.mean(
                        [test_fn(i) for i in range(n_test_batches)]))

                    print('     epoch %i, minibatch %i/%i, '
                          'test error of best model %f' %
                          (epoch, minibatch_index + 1, n_train_batches,
                          test_score))

                    # save the best model
                    with open(model_file, 'wb') as f:
                        pickle.dump(model, f)

            if patience <= iter:
                done_looping = True
                break

        # update learning rate every epoch
        model.update_learning_rate(lr_decay)
        model.update_momentum(ramp)
        print('learning rate: %f, momentum: %f' %
            (model.learning_rate, model.momentum))

    end_time = timeit.default_timer()
    print('Optimization complete with best validation score of %f, with test performance %f' % (best_validation_loss, test_score))
    print('The code run for %d epochs, with %f epochs/sec' %
        (epoch, 1. * epoch / (end_time - start_time)))
    print(('The code for file ' + os.path.split(__file__)[1] +
           ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)
