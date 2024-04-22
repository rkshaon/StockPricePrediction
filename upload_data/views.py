from django.shortcuts import render
from django.template import loader
from django.http import HttpResponse

from statsmodels.tsa.arima.model import ARIMA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



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
            df.columns = df.columns.str.strip()
            df.columns = df.columns.str.lower()
            json_data = df.to_json(orient="records")
        
            with open(f"data_collection/{company_name}.json", "w") as json_file:
                json_file.write(json_data)
                
        json_file_path = f'data_collection/{company_name}.json'
        df = pd.read_json(json_file_path)
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        train_size = int(0.8 * len(df))
        train_data, test_data = df[:train_size], df[train_size:]
        
        # arima model
        model = ARIMA(train_data['close'], order=(5, 1, 0))
        model_fit = model.fit()

        # Make predictions
        predictions = model_fit.forecast(steps=len(test_data))
        test_data['Predictions'] = predictions
        rmse = np.sqrt(
            np.mean((test_data['close'] - test_data['Predictions'])**2))

        # Plot actual vs. predicted prices
        plt.figure(figsize=(10, 6))
        plt.plot(train_data.index, train_data['close'], label='Train Data')
        plt.plot(test_data.index, test_data['close'], label='Test Data')
        plt.plot(test_data.index, test_data['Predictions'],
                label='Predictions', color='red')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.title('Stock Price Prediction using ARIMA')
        plt.legend()
        plt.show()

    context = {
        'title': 'Home | Stock Price',
    }

    template = loader.get_template('index.html')

    return HttpResponse(template.render(context, request))
