from django.db import models


class NetworkModel(models.Model):
    name = models.CharField(max_length=20, primary_key=True)
    isCreated = models.BooleanField()


class LayerType(models.Model):
    name = models.CharField(max_length=20, primary_key=True)


class NetworkLayers(models.Model):
    orderNumber = models.IntegerField(default=1, null=False)
    model = models.ForeignKey(NetworkModel, on_delete=models.CASCADE)
    layerType = models.ForeignKey(LayerType, on_delete=models.PROTECT)
    kernelAmount = models.IntegerField(default=0)
    kernelSize = models.IntegerField(default=0)
    neuronAmount = models.IntegerField(default=0)

    class Meta:
        unique_together = ('orderNumber', 'model')


class Weight(models.Model):
    name = models.CharField(max_length=20, primary_key=True, default="01")
    model = models.ForeignKey(NetworkModel, on_delete=models.CASCADE)
