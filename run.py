from neural_nets import *
from optimizers import *

# dataset
datafile = 'didi_train_rnn.h5'
with_h5f=True,
cast_y_to_int32=False

# training parameters
pretrain_epochs         = 50
n_epochs                = 1000
pretrain_learning_rate  = 0.0003
learning_rate           = 0.0003
momentum                = 0.5
lr_decay                = 0.99
ramp                    = 0.02
improvement_threshold   = 0.995
optimizer               = sgd_nesterov
model_file              = 'best_model.pkl'

# model parameters
batch_size      = 150
n_in            = 136
n_cell          = 100
n_hidden        = [1024, 256]
n_out           = 1
dropout         = [0.5, 0.5, 0.5]
chain           = 5
persistent      = False
gaussian_rbm    = True
out_function    = "relu"


def test_model_mlp():
    test_mlp(datafile=datafile, n_epochs=n_epochs, batch_size=batch_size,
        n_in=n_in, n_hidden=n_hidden, n_out=n_out, dropout=dropout,
        learning_rate=learning_rate, momentum=momentum, optimizer=optimizer,
        lr_decay=lr_decay, ramp=ramp, model_file=model_file,
        improvement_threshold=improvement_threshold, with_h5f=with_h5f,
        cast_y_to_int32=cast_y_to_int32)

def test_model_dbn():
    print('... starting dbn')
    test_dbn(
        datafile=datafile,
        pretrain_epochs=pretrain_epochs,
        n_epochs=n_epochs,
        batch_size=batch_size,
        n_visible=n_in,
        n_hidden=n_hidden,
        n_out=n_out,
        dropout=dropout,
        k=chain,
        persistent=persistent,
        gaussian_rbm=gaussian_rbm,
        out_function=out_function,
        pretrain_learning_rate=pretrain_learning_rate,
        learning_rate=learning_rate,
        momentum=momentum,
        optimizer=optimizer,
        lr_decay=lr_decay,
        ramp=ramp,
        model_file=model_file,
        improvement_threshold=improvement_threshold, with_h5f=with_h5f,
        cast_y_to_int32=cast_y_to_int32
    )
    print('dbn completed')

def test_model_rnn():
    print('... starting rnn')
    test_rnn(datafile=datafile, n_epochs=n_epochs, batch_size=batch_size,
        n_in=n_in, n_hidden=n_hidden, n_out=n_out, dropout=dropout,
        learning_rate=learning_rate, momentum=momentum, optimizer=optimizer,
        lr_decay=lr_decay, ramp=ramp, model_file=model_file,
        improvement_threshold=improvement_threshold, with_h5f=with_h5f,
        cast_y_to_int32=cast_y_to_int32)
    print('rnn completed')

def test_model_lstm():
    print('... starting lstm')
    test_rnn(datafile=datafile, n_epochs=n_epochs, batch_size=batch_size,
        n_in=n_in, n_cell=n_cell, n_hidden=n_hidden, n_out=n_out,
        dropout=dropout, learning_rate=learning_rate, momentum=momentum,
        optimizer=optimizer, lr_decay=lr_decay, ramp=ramp,
        model_file=model_file, improvement_threshold=improvement_threshold,
        with_h5f=with_h5f, cast_y_to_int32=cast_y_to_int32)
    print('lstm completed')


test_model_rnn()
print('pretrain_epochs', pretrain_epochs, 'n_epochs', n_epochs, 'pretrain_learning_rate', pretrain_learning_rate, 'learning_rate', learning_rate, 'momentum', momentum, 'lr_decay', lr_decay, 'ramp', ramp, 'improvement_threshold', improvement_threshold, 'optimizer', optimizer, 'model_file', model_file, 'batch_size', batch_size, 'n_in', n_in, 'n_cell', n_cell, 'n_hidden', n_hidden, 'n_out', n_out, 'dropout', dropout, 'chain', chain, 'persistent',  persistent, 'gaussian_rbm', gaussian_rbm, 'out_function', out_function)
