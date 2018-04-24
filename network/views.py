from django.shortcuts import render
from .forms import *
from .network_tools import *
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from .models import Weight, NetworkModel, LayerType, NetworkLayers
from keras.utils import plot_model


def network(request):
    if request.method=='GET' and request.GET:
        data = request.GET
        if data.get('edit',None)!=None:
            print('form:')
            print(data)
            print('endform')
            modelName = data['model']
            NetworkLayers.objects.filter(model__name=modelName).delete()
            i=1
            for layerType,kernel_amount,kernel_size,neuron_amount in zip(data.getlist('layerType'),data.getlist('kernelAmount'),data.getlist('kernelSize'),data.getlist('neuronAmount')):
                model = NetworkModel.objects.get(name=modelName)
                layer = LayerType.objects.get(name=layerType)
                print("model %s, layer %s" % (model,layer))
                try:
                    networkLayers = NetworkLayers.objects.create(orderNumber=i,layerType=layer,model=model,kernelAmount=kernel_amount,kernelSize=kernel_size,neuronAmount=neuron_amount)
                    print("object created")
                except Exception:
                    print("save error")
                    errorMessage = "incorrect data"
                    break
                else:
                    print("changes are saved")
                    successMessage = "changes are saved"
                i=i+1
            networkModelForm = NetworkModelForm()
        elif data.get('create',None)!=None:
            networkModelForm = NetworkModelForm(data)
            if networkModelForm.is_valid():
                print(networkModelForm.cleaned_data)
                networkModelForm.save()
                print("changes are saved")
                successMessage = "changes are saved"
            else:
                print(networkModelForm.errors)
                errorMessage = "incorrect data"

    networkLayersForm = NetworkLayersForm()
    return render(request, 'pages/network.html', locals())


def network_run(request):
    networkLayersForm = NetworkLayersForm()
    networkModelForm = NetworkModelForm()
    weightForm = WeightForm()
    return render(request,'pages/network_run.html', locals())


def model_select(request):
    returnDict=dict()
    data=request.POST
    print('model select')
    weights = Weight.objects.filter(model__name=data['model'])
    for item in weights:
        returnDict[item.name]=item.name
    print(returnDict)
    
    return JsonResponse(returnDict)
