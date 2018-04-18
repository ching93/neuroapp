from django.conf.urls import url, include
from . import views

urlpatterns = [
    url(r'^pages/', views.landing, name='pages'),
    url(r'', views.home, name='home'),
]