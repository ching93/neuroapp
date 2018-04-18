# coding: utf-8
import keras
from load_mnist import load_mnist
import model as md
import os
import numpy as np
import h5py
import json
from keras.models import Sequential
from keras.utils import plot_model

def train(model,model_name,epochs=5,input_type="2d",batch_size=128):
    (xtrain,ytrain),(xtest,ytest)=md.load_mnist(load_valid=False,input_type=input_type)
    #model = create_model(model_name)
    #history=HistoryCallback()
    model.summary()
    
    if (not model_name in os.listdir(".")):
        os.mkdir(model_name)
    os.chdir(model_name)
    plot_model(model,to_file="{0}.png".format(model_name),show_shapes=True)
    md.save_model(model_name,model)
    #return
    print("training begun")
    h=model.fit(xtrain,ytrain,epochs=epochs,batch_size=batch_size,verbose=1)
    print("training ended")
    
    model.save_weights("{}_weights.h5".format(model_name))
    
    res=open("{}_train_res.txt".format(model_name),"w")
    res.write("#, loss, acc, loss %\n")
    y=model.predict(xtest)
    pc=md.predict_percent(y,ytest)
    res.write("{0},{1},{2}\n".format(h.history["loss"],h.history["acc"],pc))
    res.close()
    os.chdir("..")

def write_percents(fname,knsize="2_2",stage=2,alpha=0.1):
    (xt,yt),(xtest,ytest) = md.load_mnist(load_valid=False)
    f=open(fname,'w')
    if (stage==2):
        for i in (4,8,16,32,64):
            for j in (4,8,16,32,64):
                model_name="model{0}_{1}-{2}".format(knsize,i,j)
                if (model_name in os.listdir(".")):
                    print(model_name)
                    f.write("{}/n".format(model_name))
                    model=md.load_model("{0}/{1}.json".format(model_name,model_name))
                    for k in range(1,5):
                        model.load_weights("{0}/{1}-{2}_weights.h5".format(model_name,model_name,k))
                        pc=md.predict_percent(model.predict(xtest),ytest,acc=alpha)
                        f.write("{}\n".format(pc))
                    f.flush()
    else:
        if (stage==1):
            for i in (4,8,16,32,64):
                model_name="model{0}_{1}".format(knsize,i)
                if (model_name in os.listdir(".")):
                    print(model_name)
                    f.write("{}/n".format(model_name))
                    model=md.load_model("{0}/{1}.json".format(model_name,model_name))
                    model.summary()
                    for k in range(1,6):
                        model.load_weights("{0}/{1}-{2}_weights.h5".format(model_name,model_name,k))
                        pc=md.predict_percent(model.predict(xtest),ytest,acc=alpha)
                        f.write("{}\n".format(pc))
                    f.flush()
        else:
            for i in (50,100,200,400,800):
                for j in (50,100,200,400,800):
                    model_name="mp_model_{1}-{2}".format(i,j)
                    if (model_name in os.listdir(".")):
                        print(model_name)
                        f.write("{}/n".format(model_name))
                        model=md.load_model("{0}/{1}.json".format(model_name,model_name))
                        for k in range(1,5):
                            model.load_weights("{0}/{1}-{2}_weights.h5".format(model_name,model_name,k))
                            pc=md.predict_percent(model.predict(xtest),ytest,acc=alpha)
                            f.write("{}\n".format(pc))
                        f.flush()
    f.close()
    
def get_summary(knsize="2",stage=1):
    #f=open(fname,'w')
    if (stage==2):
        for i in (4,8,16,32,64):
            for j in (4,8,16,32,64):
                model_name="model{0}_{1}-{2}".format(knsize,i,j)
                print(model_name)
                if (model_name in os.listdir(".")):
                    #print(model_name)
                    model=md.load_model("{0}/{1}.json".format(model_name,model_name))
                    model.summary()
    else:
        if (stage==1):
            for i in (4,8,16,32,64):
                model_name="model{0}_{1}".format(knsize,i)
                if (model_name in os.listdir(".")):
                    print(model_name)
                    model=md.load_model("{0}/{1}.json".format(model_name,model_name))
                    model.summary()
        else:
            for i in (50,100,200,400,800):
                for j in (50,100,200,400,800):
                    model_name="mp_model_{0}-{1}".format(i,j)
                    if (model_name in os.listdir(".")):
                        print(model_name)
                        model=md.load_model("{0}/{1}.json".format(model_name,model_name))
                        model.summary()
    #f.flush()
                
    
class HistoryCallback(keras.callbacks.Callback):
    def on_train_begin(self,logs={}):
        self.losses=[]
        self.total_losses=[]
    
    def on_epoch_end(self,epoch,logs={}):
        self.total_losses.append(self.losses)
        self.losses=[]
        
    def on_batch_end(self,batch,logs={}):
        self.losses.append(logs.get('loss'))
    
    
def main():
    epochs = 5
    batch_size = 16
    
    #model = md.create_model1()
    #md.save_model(model_name,model)
    #write_percents("percents_0-1.txt")
    
    kernel_sizes=[(3,3),(3,3)]
    model_name = "model3-3"
    #model = md.create_conv_model(kernel_sizes,[4,4])
    #cross_valid(model,model_name="{}_4-4".format(model_name),epochs=5)
    #model = md.create_conv_model(kernel_sizes,[4,8])
    #cross_valid(model,model_name="{}_4-8".format(model_name),epochs=5)
    #model = md.create_conv_model(kernel_sizes,[4,16])
    #cross_valid(model,model_name="{}_4-16".format(model_name),epochs=5)
    #model = md.create_conv_model(kernel_sizes,[4,32])
    #cross_valid(model,model_name="{}_4-32".format(model_name),epochs=5)
    #model = md.create_conv_model(kernel_sizes,[4,64])
    #cross_valid(model,model_name="{}_4-16".format(model_name),epochs=5)
    #model = md.create_conv_model(kernel_sizes,[8,4])
    #cross_valid(model,model_name="{}_8-4".format(model_name),epochs=5)
    #model = md.create_conv_model(kernel_sizes,[8,8])
    #cross_valid(model,model_name="{}_8-8".format(model_name),epochs=5)
    #model = md.create_conv_model(kernel_sizes,[8,16])
    #cross_valid(model,model_name="{}_8-16".format(model_name),epochs=5)
    #model = md.create_conv_model(kernel_sizes,[8,32])
    #cross_valid(model,model_name="{}_8-32".format(model_name),epochs=5)
    #model = md.create_conv_model(kernel_sizes,[8,64])
    #cross_valid(model,model_name="{}_8-64".format(model_name),epochs=5)
    #model = md.create_conv_model(kernel_sizes,[16,4])
    #cross_valid(model,model_name="{}_16-4".format(model_name),epochs=5)
    #model = md.create_conv_model(kernel_sizes,[16,8])
    #cross_valid(model,model_name="{}_16-8".format(model_name),epochs=5)
    #model = md.create_conv_model(kernel_sizes,[16,16])
    #cross_valid(model,model_name="{}_16-16".format(model_name),epochs=5)
    #model = md.create_conv_model(kernel_sizes,[16,32])
    #cross_valid(model,model_name="{}_16-32".format(model_name),epochs=5)
    #model = md.create_conv_model(kernel_sizes,[16,64])
    #cross_valid(model,model_name="{}_16-64".format(model_name),epochs=5)

    #kernel_sizes=[(4,4),(4,4)]
    #model_name = "model4-4"
    #model = md.create_conv_model(kernel_sizes,[16,4])
    #cross_valid(model,model_name="{}_16-4".format(model_name),epochs=5)
    #model = md.create_conv_model(kernel_sizes,[16,8])
    #cross_valid(model,model_name="{}_16-8".format(model_name),epochs=5)
    #model = md.create_conv_model(kernel_sizes,[16,16])
    #cross_valid(model,model_name="{}_16-16".format(model_name),epochs=5)
    #model = md.create_conv_model(kernel_sizes,[16,32])
    #cross_valid(model,model_name="{}_16-32".format(model_name),epochs=5)
    #model = md.create_conv_model(kernel_sizes,[16,64])
    #cross_valid(model,model_name="{}_16-64".format(model_name),epochs=5)
    
    
    
def one_layer_test():
    kernel_sizes=[(4,4)]
    model_name = "model4"
    for last_index in (4,8,16,32,64):
        model = md.create_conv_model(kernel_sizes,[last_index])
        cross_valid(model,model_name="{0}_{1}".format(model_name,last_index),epochs=5)
    
    kernel_sizes=[(5,5)]
    model_name = "model5"
    for last_index in (4,8,16,32,64):
        model = md.create_conv_model(kernel_sizes,[last_index])
        cross_valid(model,model_name="{0}_{1}".format(model_name,last_index),epochs=5)
    
    
def cross_valid(model,model_name,input_type='2d',epochs=5,batch_size=32):
    (xtrain,ytrain),(xvalid,yvalid),(xtest,ytest)=md.load_mnist()
    
    #model = create_model(model_name)
    #history=HistoryCallback()
    model_config = model.get_config()
    print(model.summary())
    
    if (not model_name in os.listdir(".")):
        os.mkdir(model_name)
    os.chdir(model_name)
    md.save_model(model_name,model)
    
    out_name="{0}_train_res.txt".format(model_name);
    res=open(out_name,"w")
    for i in range(1,6):
        print("\nИтерация {}".format(i))
        #model = create_model()
        
        #history=HistoryCallback()
        h=model.fit(xtrain,ytrain,epochs=epochs,batch_size=batch_size,validation_data=(xvalid,yvalid))
        
        swap_intl=range((i-1)*10000,(i*10000-1))
        swap=(xtrain[swap_intl],ytrain[swap_intl])
        (xtrain[swap_intl],ytrain[swap_intl])=(xvalid[0:9999],yvalid[0:9999])
        (xvalid,yvalid)=swap
        
        model.save_weights("{}-{}_weights.h5".format(model_name,i))
        
        print("evaluating...")
        score=model.evaluate(xtest,ytest)
        
        res.write("#, vloss, vacc, tloss, tvacc, %\n")
        y=model.predict(xtest)
        pc=md.predict_percent(y,ytest)
        res.write("{0},{1},{2},{3},{4},{5}\n".format(i,h.history["val_loss"],h.history["val_acc"],score[0],score[1],pc))
        res.flush()
        del model
        model = Sequential.from_config(model_config)
        model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
    res.close()
    os.chdir("..")

def test():
    model = md.create_model4()
    mnist = h5py.File("mnist.hdf5")
    (xtest,ytest) = (mnist["test/images"], mnist["test/labels"])
    xtest=format_images(xtest[0:],'2d')
    ytest=keras.utils.to_categorical(ytest)
    #for i in range(1,6):
    #    model.load_weights("model1-{}_weights.h5".format(i))
    #    y = model.predict(xtest)
    #    print("{0}я модель: {1}%".format(i,md.predict_percent(y,ytest)))
    model.load_weights("model_weights/model4-5_weights.h5")
    y = model.predict(xtest)
    i=0

    print("{0}я модель: число ошибок {1}".format(i,md.predict_percent(y,ytest)))
    

def predict():
    (xt,yt),(xt1,yt1)=load_mnist()
    xt1=xt1.reshape(xt1.shape[0],784)
    model=md.load_model('model3.json')
    #model=md.create_model4()
    model.load_weights('model3_weights.h5')
    y=model.predict(xt1)
    print(predict_percent(y,yt1))


#get_summary("-",stage=3)
get_summary("2",stage=1)
get_summary("3",stage=1)
get_summary("4",stage=1)
get_summary("5",stage=1)
#main()
#one_layer_test()
#write_percents("all_percents(2).txt",knsize="2",stage=1)
#write_percents("all_percents(3).txt",knsize="3",stage=1)
#write_percents("all_percents(4).txt",knsize="4",stage=1)
#test()
              #predict()
#cross_valid(md.create_model6,"model6")
#cross_valid1(md.create_model3,"model3")
