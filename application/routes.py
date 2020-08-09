from flask import request, render_template
from application.preprocessing import DataPreprocessing
from sklearn.ensemble import RandomForestRegressor
from application import application
import numpy as np
import pandas as pd
import pickle


@application.route('/')
def home():
    return render_template('index.html')


@application.route('/predict', methods=['POST'])
def predict():
    # For rendering results on HTML GUI

    features = np.array([x for x in request.form.values()])
    features = features.reshape(1,-1)
    # Sample Feature Dataset to make the test_df Dataframe in sync with feature_dataset

    features_dataset = pd.read_csv("https://github.com/KarthikChandrashekar5230/FlightDelay-Prediction-Flask/blob/master/application/Features_Dataset.csv?raw=true", header=0, low_memory=False)
    features_dataset.drop(['Unnamed: 0'], axis=1, inplace=True)

    # Features required for prediction
    columns = ['MONTH', 'DAY_OF_WEEK', 'SCHEDULED_TIME', 'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
               'SCHEDULED_DEPARTURE',
               'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']

    test_df = pd.DataFrame(data=features, columns=columns)
    data_preprocess = DataPreprocessing()
    test_df = data_preprocess.features_dataype_check(test_df)
    test_df = data_preprocess.data_preprocessing(test_df)
    test_df = data_preprocess.create_dummyvariables(test_df)
    rf_model = pickle.load(open('application/RFRegressor-65MB.pkl', 'rb'))

    # Sample Feature Dataset to make the test_df Dataframe in sync with feature_dataset
    test_df = features_dataset.append([test_df])
    test_df.fillna(0, inplace=True)

    # Predict the delay
    prediction=rf_model.predict(np.array(test_df).reshape(1,-1))
    prediction=round(prediction[0],3)

    return render_template('index.html', prediction_text='Expected Flight Delay would be(in mins): {}'.format(prediction))
