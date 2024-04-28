from django.shortcuts import render
from django.template import loader
from django.http import HttpResponse

from statsmodels.tsa.arima.model import ARIMA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
# from fbprophet import Prophet
from prophet import Prophet



def index(request):
    context = {
        'title': 'Home | Stock Price',
    }

    if request.method == 'POST':
        company_name = request.POST.get('company')
        file = request.FILES.get('file')

        if not company_name or not file:
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
        
        # Prophet
        last_date = df.iloc[0]['date'].date()
        # Prepare the data
        df.rename(columns={'date': 'ds', 'close': 'y'}, inplace=True)

        # Create and fit the Prophet model
        model = Prophet()
        model.fit(df)

        # Make predictions for the next 365 days
        future = model.make_future_dataframe(periods=365)
        forecast = model.predict(future)

        data = []
        
        for i in range(0, 5):
            temp = {
                'date': (last_date + pd.Timedelta(days=(i+1))),
            }
            
            try:
                temp['close'] = round(forecast[forecast['ds'] == str(temp['date'])]['yhat'].values[0], 2)
            except Exception as e:
                temp['close'] = 0
                
            data.append(temp)
            
        context['predicted_data'] = data
        context['company_name'] = company_name
        
        # print(last_date)
        # print(forecast)
        # print(forecast[])
        # specific_date = '2024-04-27'
        # prediction_for_specific_date = forecast[forecast['ds'] == specific_date]['yhat'].values[0]
        # print(specific_date, prediction_for_specific_date)
        # # Visualize the forecast
        # fig = model.plot(forecast)
        # print(fig)


        # # LSTM
        # df['date'] = pd.to_datetime(df['date'])
        # df.set_index('date', inplace=True)
        # scaler = MinMaxScaler()
        # df['close_scaled'] = scaler.fit_transform(df[['close']])

        # # Create sequences for training
        # sequence_length = 60  # Number of days to look back
        # X, y = [], []
        # for i in range(len(df) - sequence_length):
        #     X.append(df['close_scaled'].iloc[i:i+sequence_length])
        #     y.append(df['close_scaled'].iloc[i+sequence_length])

        # X, y = np.array(X), np.array(y)
        # train_size = int(0.8 * len(X))

        # X_train, y_train = X[:train_size], y[:train_size]
        # X_test, y_test = X[train_size:], y[train_size:]

        # # Build the LSTM model
        # model = Sequential()
        # model.add(LSTM(50, input_shape=(sequence_length, 1)))
        # model.add(Dense(1))
        # model.compile(optimizer='adam', loss='mean_squared_error')

        # # Train the model
        # model.fit(X_train, y_train, epochs=10, batch_size=32)

        # # Evaluate the model on test data
        # test_loss = model.evaluate(X_test, y_test)
        # # print(f"Test Loss: {test_loss:.4f}")

        # # Make predictions
        # predictions = model.predict(X_test)
        # predicted_prices = scaler.inverse_transform(predictions)
        
        # print(predicted_prices[0][0])
        # print(predicted_prices[1][0])
        # print(predicted_prices[2][0])
        # print(predicted_prices[3][0])
        # print(predicted_prices[4][0])
        
        # data = []
        # last_date = df.index[0]

        # for i in range(0, 5):
        #     data.append({
        #         'date': (last_date + pd.Timedelta(days=(i+1))).date(),
        #         'close': round(predicted_prices[i][0], 2),
        #     })
            
        # context['predicted_data'] = data
        # context['company_name'] = company_name

    template = loader.get_template('index.html')

    return HttpResponse(template.render(context, request))
