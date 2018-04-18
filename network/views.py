from django.shortcuts import render
from .forms import *
# import network.model as m
from .network_tools import *
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from .models import Weight, NetworkModel, LayerType, NetworkLayers
from keras.utils import plot_model

def network(request):
    if request.method=='GET' and request.GET:
        data = request.GET
        print('form:')
        print(data)
        print('endform')
        modelName = data['model']
        NetworkLayers.objects.filter(model__name=modelName).delete()
        i=1
        for layerType,kernel_amount,kernel_size,neuron_amount in zip(data.getlist('layerType'),data.getlist('kernelAmount'),data.getlist('kernelSize'),data.getlist('neuronAmount')):
            #layerType = request.POST['layerType[]']
            print('layerType=%s,kernel_amount=%s,kernel_size=%s,neuron_amount=%s' % (layerType,kernel_amount,kernel_size,neuron_amount))
            model = NetworkModel.objects.get(name=modelName)
            layer = LayerType.objects.get(name=layerType)
            print("model %s, layer %s" % (model,layer))
            try:
                networkLayers = NetworkLayers.objects.create(orderNumber=i,layerType=layer,model=model,kernelAmount=kernel_amount,kernelSize=kernel_size,neuronAmount=neuron_amount)
                print("object created")
            except Exception:
                print("ошибка сохранения")
                errorMessage = "Неправильные данные"
                break
            else:
                print("изменения сохранены")
                successMessage = "Изменения сохранены"
            i=i+1
    networkLayersForm = NetworkLayersForm()
    return render(request, 'pages/network.html', locals())


def network_run(request):
    networkLayersForm = NetworkLayersForm()
    networkModelForm = NetworkModelForm()
    weightForm = WeightForm()
    return render(request,'pages/network_run.html', locals())

# def train(model, model_name, epochs=5, input_type="2d", batch_size=128):
def network_train(request):
    returnDict=dict()
    data=request.POST
    print(data)
    if data['action']=='model-select':
        print('model select')
        weights = Weight.objects.filter(model__name=data['model'])
        print(weights)
        for item in weights:
            returnDict[item.name]=item.name
        print(returnDict)
    elif data['action']=='train':
        print('model create')
        try:
            (model,input_type) = createNetwork(data["model"])
        except Exception:
            returnDict["success"] = 0
            returnDict["message"] = "Ошибка создания модели"
            print("Ошибка создания модели")
            return JsonResponse(returnDict)

        epochs = int(data['epochs'])
        batch_size = int(data['batch-size'])
        interval = [int(data['from']),int(data['till'])]
        print(interval)
        try:
            (y,pc) = train(model,data["model"],epochs=epochs,interval=interval,input_type=input_type, batch_size=batch_size)
        except Exception:
            returnDict["success"] = 0
            returnDict["message"] = "Ошибка во время обучения"
            print("Ошибка во время обучения")
            return JsonResponse(returnDict)

        returnDict['accurasy']=y
        returnDict['percent']=pc

    elif data['action']=='state-check':
        print('check')
        returnDict['loss']=common_data.loss
    
    returnDict["success"] = 1
    returnDict["message"] = "Обучение завершено"
    return JsonResponse(returnDict)




#class TestRoute(WebSocketRoute):
#    # This method will be executed when client will call route-alias first time.
#    def init(self, **kwargs):
#        # the python __init__ must be return "self". This method might return anything.
#        print("connected")
#        return kwargs
#    def do_notify(self):
#
#        yield self.socket.call('progress_info',result="hello from python!")
#
#def init_socket():
#    WebSocket.ROUTES['test'] = TestRoute
#    WebSocket.init_pool()