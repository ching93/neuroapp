from .nn_model import create_conv_model, create_mp_model, predict_percent
from .models import *
import os
from .nn_model import format_images, load_mnist, save_model, predict_percent, load_model, average_error, save_plot
from keras.utils import plot_model
from keras.callbacks import Callback
import json


def create_network(network_name):
    layers = NetworkLayers.objects.filter(model__name=network_name)
    kernel_sizes = []
    kernel_amounts = []
    pooling_sizes = []
    dense_layer_sizes = []
    layers.order_by('orderNumber')
    for layer in layers:
        if layer.layerType.name == 'Convolutional':
            kernel_sizes.append([layer.kernelSize, layer.kernelSize])
            kernel_amounts.append(layer.kernelAmount)
        elif layer.layerType.name == 'Pooling':
            pooling_sizes.append([layer.kernelSize, layer.kernelSize])
        elif layer.layerType.name == 'Dense':
            dense_layer_sizes.append(layer.neuronAmount)
    print("created network info:")
    print("kernel_sizes={}".format(kernel_sizes))
    print("kernel_amounts={}".format(kernel_amounts))
    print("pooling_sizes={}".format(pooling_sizes))
    print("dense_layer_sizes={}".format(dense_layer_sizes))
    if kernel_sizes != []:
        model = create_conv_model(kernel_sizes=kernel_sizes, kernel_nums=kernel_amounts,
                                  pool_sizes=pooling_sizes, dense_layer_sizes=dense_layer_sizes)
    else:
        model = create_mp_model(dense_layer_sizes)

    model.summary()
    #plot_model(model, to_file="static/images/model_scheme.png", show_shapes=True)

    if (len(kernel_sizes) > 0):
        input_type = "2d"
    else:
        input_type = "1d"
    return (model, input_type)


def train(model, model_name, interval=[1, 100], epochs=5, input_type="2d", batch_size=128, socket=None, save_weights=False):
    (xtrain, ytrain), (xtest, ytest) = load_mnist(load_valid=False,
                                                  path='static/media/mnist.hdf5', input_type=input_type)
    print("training begun")
    if socket is not None:
        callback = Loss_history(socket)
    xtrain = xtrain[range(interval[0], interval[1])]
    ytrain = ytrain[range(interval[0], interval[1])]
    ratio = (interval[1]-interval[0])/6000
    b = round(ratio*10000)
    xtest = xtest[range(1, b)]
    ytest = ytest[range(1, b)]

    h = model.fit(xtrain, ytrain, epochs=epochs,
                  batch_size=batch_size, verbose=0, callbacks=[callback])
    print("training ended")

    if (save_weights):
        print('saving weights')
        model.save_weights("{0}{1}_weights.h5".format(file_root, model_name))

    print("model predicting")
    y = model.predict(xtest)
    av_err = float(average_error(y, ytest))
    pc = float(predict_percent(y, ytest))
    return (av_err, pc)


class Loss_history(Callback):
    def __init__(self, socket):
        self.socket = socket
        self.loss = []
        self.acc = []

    def on_train_begin(self, logs={}):
        print("обучение началось")
        data = {'action': 'train-begin'}
        self.socket.send(json.dumps(data))

    def on_epoch_begin(self, epoch, logs={}):
        print('%dth epoch: %s' % (epoch, logs))
        data = {'action': 'epoch-begin', 'epoch': epoch+1}
        self.socket.send(json.dumps(data))

    def on_batch_end(self, batch, logs={}):
        print('%dth batch: %s' % (batch, logs))
        loss = float(logs.get('loss'))
        acc = float(logs.get('acc'))
        self.loss.append(loss)
        self.acc.append(acc)
        save_plot(self.loss, self.acc, "static/images", "train_pic.jpg")

        data = {'action': 'batch-end',
                'batch': batch+1, 'loss': loss, 'acc': acc}
        self.socket.send(json.dumps(data))

    def on_train_end(self, log={}):
        print("saving")
        save_plot(self.loss, self.acc, "static/images", "train_pic.jpg")
        data = {'action': 'train-end'}
        self.socket.send(json.dumps(data))
