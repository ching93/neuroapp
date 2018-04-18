from django.db import models


class NetworkModel(models.Model):
    name = models.CharField(max_length=20, primary_key=True)
    isCreated = models.BooleanField()

    class Meta:
        verbose_name = "Модель НС"
        verbose_name_plural = "Модели НС"


class LayerType(models.Model):
    name = models.CharField(max_length=20, primary_key=True)
    class Meta:
        verbose_name = "Тип слоя"
        verbose_name_plural = "Типы слоев"


class NetworkLayers(models.Model):
    orderNumber = models.IntegerField(default=1, null=False)
    model = models.ForeignKey(NetworkModel, on_delete=models.CASCADE)
    layerType = models.ForeignKey(LayerType, on_delete=models.PROTECT)
    kernelAmount = models.IntegerField(default=0)
    kernelSize = models.IntegerField(default=0)
    neuronAmount = models.IntegerField(default=0)


    class Meta:
        verbose_name = "Информация о слое"
        verbose_name_plural = "Информация о слоях"
        unique_together = ('orderNumber','model')


class Weight(models.Model):
    name = models.CharField(max_length=20, primary_key=True,default="01")
    model = models.ForeignKey(NetworkModel, on_delete=models.CASCADE)
    class Meta:
        verbose_name = "Весовые коэффициенты"
        verbose_name_plural = "Весовые коэффициенты"



# class DataSet(models.Model):
#     name = models.CharField(max_length=20, primary_key=True)
#     file = models.FileField(upload_to="network_models/")
#
#     class Meta():
#         verbose_name = "Набор данных"
#         verbose_name_plural = "Наборы данных"
