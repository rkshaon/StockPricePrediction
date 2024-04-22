from django.shortcuts import render
from django.template import loader
from django.http import HttpResponse



def index(request):
    if request.method == 'POST':
        file = request.FILES.get('file')

        if file and file.content_type == 'text/csv':
            print('format...')

        # # Read the uploaded file into a Pandas DataFrame
        # import pandas as pd
        # df = pd.read_csv(uploaded_file)

        # # Convert DataFrame to JSON
        # json_data = df.to_json(orient="records")

        # # Save JSON data to a file (e.g., "uploaded_data.json")
        # with open("uploaded_data.json", "w") as json_file:
        #     json_file.write(json_data)

    context = {
        'title': 'Home | Stock Price',
    }

    template = loader.get_template('index.html')

    return HttpResponse(template.render(context, request))
