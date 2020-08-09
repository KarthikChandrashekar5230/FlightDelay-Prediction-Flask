import pandas as pd
#import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from application.model_prep import ModelBuilding


class DataPreprocessing:

    def create_raw_dataframe(self,filepath):

        dataframe = pd.read_csv(filepath, header=0,low_memory=False)

        return dataframe

    def features_dataype_check(self,raw_dataframe):

        raw_dataframe[['MONTH', 'DAY_OF_WEEK', 'SCHEDULED_DEPARTURE']] = raw_dataframe[['MONTH', 'DAY_OF_WEEK', 'SCHEDULED_DEPARTURE']].astype('int64')
        raw_dataframe[['SCHEDULED_TIME', 'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']] = raw_dataframe[['SCHEDULED_TIME',
             'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']].astype('float32')
        raw_dataframe[['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']] = raw_dataframe[['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']].astype('str')

        return raw_dataframe

    def data_preprocessing(self,raw_dataframe):

        raw_dataframe['ORIGIN_AIRPORT'] = raw_dataframe['ORIGIN_AIRPORT'].apply(lambda x: re.sub('[0-9]+', 'OTH-O', x))
        raw_dataframe['DESTINATION_AIRPORT'] = raw_dataframe['DESTINATION_AIRPORT'].apply(lambda x: re.sub('[0-9]+', 'OTH-D', x))
        raw_dataframe['AIRLINE'] = raw_dataframe['AIRLINE'].str.upper()
        raw_dataframe['ORIGIN_AIRPORT'] = raw_dataframe['ORIGIN_AIRPORT'].str.upper()
        raw_dataframe['DESTINATION_AIRPORT'] = raw_dataframe['DESTINATION_AIRPORT'].str.upper()
        
        raw_dataframe.fillna(0, inplace=True)

        return raw_dataframe

    def avoid_irrelevant_features(self,raw_dataframe):

        raw_dataframe.drop(['Unnamed: 0', 'YEAR', 'DAY', 'FLIGHT_NUMBER', 'TAIL_NUMBER', 'WHEELS_OFF', 'Unnamed: 0.1',
                      'ELAPSED_TIME', 'AIR_TIME', 'DISTANCE', 'WHEELS_ON', 'TAXI_IN', 'SCHEDULED_ARRIVAL',
                      'ARRIVAL_TIME',
                      'DEPARTURE_TIME', 'ARRIVAL_DELAY', 'DIVERTED', 'CANCELLED', 'CANCELLATION_REASON', 'TAXI_OUT'],
                     axis=1, inplace=True)

        return raw_dataframe

    def rearranging_columns(self,raw_dataframe):

        raw_dataframe = raw_dataframe[['DEPARTURE_DELAY', 'MONTH', 'DAY_OF_WEEK', 'SCHEDULED_TIME', 'AIRLINE', 'ORIGIN_AIRPORT',
                           'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE',
                           'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY',
                           'WEATHER_DELAY']]

        return raw_dataframe

    def create_dummyvariables(self,raw_dataframe):

        def time_format_conversion(time):
            time = int(time / 100)

            return time

        raw_dataframe = pd.get_dummies(raw_dataframe, columns=['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT'])
        raw_dataframe['SCHEDULED_DEPARTURE'] = raw_dataframe['SCHEDULED_DEPARTURE'].apply(time_format_conversion)

        return raw_dataframe

    def features_target_selection(self,raw_dataframe):

        features = raw_dataframe.columns[1:]
        target = raw_dataframe.columns[0]

        return raw_dataframe,features,target

    def dataset_split(self,raw_dataframe,features,target):

        input_train, input_test, target_train, target_test = train_test_split(raw_dataframe[features],
                                                                              raw_dataframe[target], test_size=0.2,
                                                                              random_state=42)
        return input_train, input_test, target_train, target_test


if __name__ == "__main__":

    preprocess=DataPreprocessing()

    raw_dataframe = preprocess.create_raw_dataframe("C:\\Users\\kp\\Pictures\\Air Flight Project\\Data\\Shuffled_Data.csv")
    raw_dataframe = preprocess.features_dataype_check(raw_dataframe)
    raw_dataframe = preprocess.data_preprocessing(raw_dataframe)
    raw_dataframe = preprocess.avoid_irrelevant_features(raw_dataframe)
    raw_dataframe = preprocess.rearranging_columns(raw_dataframe)
    raw_dataframe = preprocess.create_dummyvariables(raw_dataframe)
    processed_dataframe,features,target = preprocess.features_target_selection(raw_dataframe)
    input_train, input_test, target_train, target_test = preprocess.dataset_split(processed_dataframe,features,target)

    modelbuilding = ModelBuilding()

    #modelbuilding.randonmizedgridsearchCV_results()
    modelbuilding.modelbuilding_performance(input_train, input_test, target_train, target_test)
