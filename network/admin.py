from django.contrib import admin
from .models import *


class NetworkModelAdmin(admin.ModelAdmin):
    list_display = [field.name for field in NetworkModel._meta.fields]

    class Meta:
        model = NetworkModel


class LayerTypeAdmin(admin.ModelAdmin):
    list_display = [field.name for field in LayerType._meta.fields]

    class Meta:
        model = LayerType


class NetworkLayersAdmin(admin.ModelAdmin):
    list_display = [field.name for field in NetworkLayers._meta.fields]

    class Meta:
        model = NetworkLayers


class WeightAdmin(admin.ModelAdmin):
    list_display = [field.name for field in Weight._meta.fields]

    class Meta:
        model = Weight


admin.site.register(NetworkModel, NetworkModelAdmin)
admin.site.register(LayerType, LayerTypeAdmin)
admin.site.register(NetworkLayers, NetworkLayersAdmin)
admin.site.register(Weight, WeightAdmin)