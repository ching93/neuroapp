# coding: utf-8
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import model_from_json
import os
#from keras.utils import plot_model
import numpy as np
import h5py

def create_model1():
    input_shape = (28,28,1)
    num_classes = 10

    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
    return model

def create_model2():
    input_shape = (28,28,1)
    num_classes = 10

    model = Sequential()
    model.add(Conv2D(16, kernel_size=(2, 2),
                 activation='relu',
                 input_shape=input_shape))
    model.add(Conv2D(32, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
    #plot_model(model,'model2.png')
    return model
def create_model3():
    input_shape = (784,)
    num_classes = 10

    model = Sequential()
    model.add(Dense(100, activation='relu', input_shape=input_shape))
    model.add(Dense(400, activation='relu'))
    model.add(Dense(600, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
    plot_model(model,'model3.png')
    return model

def create_model4():
    input_shape = (28,28,1)
    num_classes = 10

    model = Sequential()
    initializer = keras.initializers.RandomUniform(minval=-0.5,maxval=0.5)
    model.add(Conv2D(16, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape,kernel_initializer=initializer))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu',kernel_initializer=initializer))
    model.add(Flatten())
    model.add(Dense(128, activation='relu',kernel_initializer=initializer))
    model.add(Dense(num_classes, activation='softmax',kernel_initializer=initializer))

    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
    #plot_model(model,'model4.png')
    return model
def create_model5():
    input_shape = (28,28,1)
    num_classes = 10

    model = Sequential()
    initializer = keras.initializers.RandomUniform(minval=-0.5,maxval=0.5)
    model.add(Conv2D(16, kernel_size=(2, 2),
                 activation='relu',
                 input_shape=input_shape,kernel_initializer=initializer))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu',kernel_initializer=initializer))
    model.add(Flatten())
    model.add(Dense(128, activation='relu',kernel_initializer=initializer))
    model.add(Dense(num_classes, activation='softmax',kernel_initializer=initializer))

    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
    return model
def create_model6():
    input_shape = (28,28,1)
    num_classes = 10

    model = Sequential()
    initializer = keras.initializers.RandomUniform(minval=-0.5,maxval=0.5)
    model.add(Conv2D(16, kernel_size=(2, 2),
                 activation='relu',
                 input_shape=input_shape,kernel_initializer=initializer))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (2, 2), activation='relu',kernel_initializer=initializer))
    model.add(Flatten())
    model.add(Dense(128, activation='relu',kernel_initializer=initializer))
    model.add(Dense(num_classes, activation='softmax',kernel_initializer=initializer))

    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
    plot_model(model,'model6.png')
    return model

def create_model7(model_name):
    input_shape = (28,28,1)
    num_classes = 10

    model = Sequential()
    initializer = keras.initializers.RandomUniform(minval=-0.5,maxval=0.5)
    model.add(Conv2D(16, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape,kernel_initializer=initializer))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu',kernel_initializer=initializer))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu',kernel_initializer=initializer))
    model.add(Dense(64, activation='relu',kernel_initializer=initializer))
    model.add(Dense(num_classes, activation='softmax',kernel_initializer=initializer))

    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
    save_model(model_name,model)
    #plot_model(model,'model6.png')
    return model

def create_conv_model(kernel_sizes,kernel_nums,pool_sizes=[(2,2)],dense_layer_sizes=[128,64]):
    input_shape = (28,28,1)
    num_classes = 10
    conv_layers=len(kernel_sizes)
    if (len(pool_sizes)==1):
        pool_sizes=[pool_sizes[0] for i in range(0,conv_layers)]
    if (len(kernel_sizes)!=conv_layers or len(pool_sizes)!=conv_layers):
        raise NameError("число наборов ядер не равно числу слоев")
    initializer = keras.initializers.RandomUniform(minval=-0.5,maxval=0.5)
        
    model = Sequential()
    model.add(Conv2D(kernel_nums[0],kernel_size=kernel_sizes[0],activation='relu',input_shape=input_shape,kernel_initializer=initializer))
    model.add(MaxPooling2D(pool_size=pool_sizes[0]))
    for i in range(1,conv_layers):
        model.add(Conv2D(kernel_nums[i],kernel_size=kernel_sizes[i],activation='relu',kernel_initializer=initializer))
        model.add(MaxPooling2D(pool_size=pool_sizes[i]))
    model.add(Flatten())
    for i in range(0,len(dense_layer_sizes)):
        model.add(Dense(dense_layer_sizes[i],activation='relu',kernel_initializer=initializer))
    model.add(Dense(num_classes,activation='softmax',kernel_initializer=initializer))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
    
    return model

def create_mp_model(layer_sizes):
    input_shape = (784,)
    num_classes = 10
    mp_layers=len(layer_sizes)
    
    model = Sequential()
    initializer = keras.initializers.RandomUniform(minval=-0.5,maxval=0.5)
    model.add(Dense(layer_sizes[0],activation='relu', input_shape=input_shape, kernel_initializer=initializer))
    for i in range(1,len(layer_sizes)):
        model.add(Dense(layer_sizes[i], activation='relu', kernel_initializer=initializer))
    model.add(Dense(num_classes,activation='softmax',kernel_initializer=initializer))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
    
    return model

def load_mnist(load_valid=True,path="mnist.hdf5",normalize=True,input_type='2d'):
    files = h5py.File(path)
    xtrain = files["train/images"][0:].astype('float32')
    ytrain = keras.utils.to_categorical(files["train/labels"][0:])
    xtest = files["test/images"][0:].astype('float32')
    ytest = keras.utils.to_categorical(files["test/labels"][0:])
    xtrain = format_images(xtrain,input_type)
    xtest = format_images(xtest,input_type)
    
    if (normalize):
        xtrain /= 255
        xtest /= 255
    if (load_valid):
        return (xtrain[0:49999],ytrain[0:49999]),(xtrain[50000:],ytrain[50000:]),(xtest,ytest)
    else:
        return (xtrain,ytrain),(xtest,ytest)

def save_model(fname,model):
    with open('{}.json'.format(fname),'w') as json_file:
        json_file.write(model.to_json())
    #plot_model(model,'{}.png'.format(fname),show_shapes=True)
    print('model saved')
    
def evaluate(model,xtest,ytest):
    y = model.predict(xtest)
    ##err = np.sum((y-ytest)**2)/y.shape[0]
    score = keras.losses.categorical_crossentropy(y,ytest)
    percent = predict_percent(y,ytest)
    return (score[0],score[1],percent)

def load_model(fname):
    json_file = open(fname,'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    print('model loaded')
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
    return model

def predict_percent(y,label,acc=0.01):
    n=y.shape[0]
    m=0.0
    for i in range(1,y.shape[0]):
        max_index=np.argmax(y[i])
        dy=np.abs(y[i]-label[i])
        for j in range(0,len(dy)):
            if (dy[j]>acc):
                m+=1
                break
    return m/n*100

def save_result(fname,model_name,result):
    f = open(fname,'a')
    f.write("Model {0}:\n{1}\n".format(model_name,result))
    f.close()
    
def format_images(images,format_type="2d"):
    if (format_type=='2d'):
        return images.reshape(images.shape[0],28,28,1)
    else:
        return images.reshape(images.shape[0],28*28)
