# Stock Price Prediction
This repository take input a Company's stock price history data and return predicted result using `Prophet`, `LSTM` and `ARIMA` models.

## Setup and Installation
Open the terminal on the directory where you want to run, and then clone the project using the command below:
```
git clone git@github.com:rkshaon/StockPricePrediction.git
```

Create and activate the virtual environment

[Create/activate Virtual Environment on Windows or Unix/MacOS](https://github.com/rkshaon/software-engineering-preparation/tree/master/Languages/Python/environment)

After activating the virtual environment, install the dependecies using the command below:
```
pip install -r requirements.txt
```

In this project there no model is used to store data, so `migrate`, and also `superuser` will not be needed.

Run the project
```
python manage.py runserver
```

This will run the project on **localhost:8000** or **127.0.0.1:8000** address. Open this address on your web browser.