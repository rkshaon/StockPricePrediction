from django.shortcuts import render
from django.template import loader
from django.http import HttpResponse



def index(request):
    context = {
        'title': 'Home | Stock Price',
    }

    template = loader.get_template('index.html')

    return HttpResponse(template.render(context, request))
