from django.shortcuts import render
from django.template import loader
from django.http import HttpResponse

from statsmodels.tsa.arima.model import ARIMA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from prophet import Prophet

from upload_data.utility import get_real_data



def index(request):
    context = {
        'title': 'Home | Stock Price',
    }

    if request.method == 'POST':
        company_name = request.POST.get('company')
        company_code = request.POST.get('company_code')
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

        context['company_name'] = company_name
        json_file_path = f'data_collection/{company_name}.json'
        current_date = datetime.date.today()

        # Prophet
        df = pd.read_json(json_file_path)
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
        sd = last_date + pd.Timedelta(days=1)
        ed = sd + datetime.timedelta(days=7)
        weekends = {4, 5}

        while sd != ed:
            if sd.weekday() not in weekends:
                temp = {
                    'date': sd,
                    'close': round(forecast[forecast['ds'] == str(sd)]['yhat'].values[0], 2),
                    # 'real': None,
                }

                # if current_date > sd:
                #     print(current_date, sd, 'Get this data')
                #     get_real_data(company_code=company_code)
                #     temp['real'] = 'coming-soon'
                
                data.append(temp)

            sd += datetime.timedelta(days=1)
            
        context['prophet_predicted_data'] = data

        # LSTM
        df = pd.read_json(json_file_path)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        scaler = MinMaxScaler()
        df['close_scaled'] = scaler.fit_transform(df[['close']])

        # Create sequences for training
        sequence_length = 60  # Number of days to look back
        X, y = [], []
        for i in range(len(df) - sequence_length):
            X.append(df['close_scaled'].iloc[i:i+sequence_length])
            y.append(df['close_scaled'].iloc[i+sequence_length])

        X, y = np.array(X), np.array(y)
        train_size = int(0.8 * len(X))

        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]

        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(50, input_shape=(sequence_length, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=32)

        # Evaluate the model on test data
        test_loss = model.evaluate(X_test, y_test)
        # print(f"Test Loss: {test_loss:.4f}")

        # Make predictions
        predictions = model.predict(X_test)
        predicted_prices = scaler.inverse_transform(predictions)
        
        data = []
        last_date = df.index[0]
        sd = last_date + pd.Timedelta(days=1)
        ed = sd + datetime.timedelta(days=7)
        weekends = {4, 5}
        i = 0

        while sd != ed:
            if sd.weekday() not in weekends:
                data.append({
                    'date': sd.date(),
                    'close': round(predicted_prices[i][0], 2),
                })

            i += 1
            sd += datetime.timedelta(days=1)
            
        context['lstm_predicted_data'] = data

        # ARIMA
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
        
        predicted_values = model_fit.forecast(steps=7)

        # Extract the predicted stock prices
        predicted_prices = predicted_values.tolist()

        data = []
        last_date = df.index[0]
        last_date = last_date.date()
        sd = last_date + pd.Timedelta(days=1)
        ed = sd + datetime.timedelta(days=7)
        weekends = {4, 5}
        i = 0
        
        while sd != ed:
            if sd.weekday() not in weekends:
                data.append({
                    'date': sd,
                    'close': round(predicted_prices[i], 2),
                })

            i += 1
            sd += datetime.timedelta(days=1)
            
        context['arima_predicted_data'] = data

    template = loader.get_template('index.html')

    return HttpResponse(template.render(context, request))
