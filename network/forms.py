from django import forms
from .models import NetworkModel, LayerType, NetworkLayers, Weight


class NetworkModelForm(forms.ModelForm):

    class Meta:
        model = NetworkModel
        exclude = ["created"]


class LayerTypeForm(forms.ModelForm):

    class Meta:
        model = LayerType
        exclude = [""]


class NetworkLayersForm(forms.ModelForm):
    class Meta:
        model = NetworkLayers
        exclude = [""]


class WeightForm(forms.ModelForm):

    class Meta:
        model = Weight
        exclude = [""]
