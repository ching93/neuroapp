from django.conf.urls import url, include
from . import views

urlpatterns = [
    url(r'^network$', views.network, name='network_edit'),
    url(r'^network_run$', views.network_run, name='network_run'),
]