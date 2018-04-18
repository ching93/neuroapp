from django.shortcuts import render
from .forms import *
#import network.program_mnist as pm
#import libs.model as m


def landing(request):
    form = SubscriberForm(request.POST or None)

    if request.method == "POST" and form.is_valid():
        print(request.POST)
        new_from = form.save()

    return render(request, 'pages/landing.html', locals())

def home(request):

    return render(request, 'pages/home.html', locals())

