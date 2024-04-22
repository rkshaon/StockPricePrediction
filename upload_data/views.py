from django.shortcuts import render
from django.template import loader
from django.http import HttpResponse

import pandas as pd



def index(request):
    if request.method == 'POST':
        company_name = request.POST.get('company')
        file = request.FILES.get('file')

        if not company_name or not file:
            context = {
                'title': 'Home | Stock Price',
            }

            template = loader.get_template('index.html')

            return HttpResponse(template.render(context, request))

        if file and file.content_type == 'text/csv':
            df = pd.read_csv(file)
            json_data = df.to_json(orient="records")
        
            with open(f"data_collection/{company_name}.json", "w") as json_file:
                json_file.write(json_data)

    context = {
        'title': 'Home | Stock Price',
    }

    template = loader.get_template('index.html')

    return HttpResponse(template.render(context, request))
