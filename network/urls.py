from django.conf.urls import url
from . import views


urlpatterns = [
    url(r'^network$', views.network, name='network_edit'),
    url(r'^network_run$', views.network_run, name='network_run'),
    url(r'^model_select$', views.model_select, name='model_select'),
]