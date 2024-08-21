import pickle
import inflection
import pandas as pd
import numpy as np
import math
import datetime

class Rossmann(object):
    def __init__(self):
        self.home_path = ''
        self.competition_distance_scaler   = pickle.load( open( self.home_path + 'parameter/competition_distance_scaler.pkl', 'rb') )
        self.competition_time_month_scaler = pickle.load( open( self.home_path + 'parameter/competition_time_month_scaler.pkl', 'rb') )
        self.promo_time_week_scaler        = pickle.load( open( self.home_path + 'parameter/promo_time_week_scaler.pkl', 'rb') )
        self.year_scaler                   = pickle.load( open( self.home_path + 'parameter/year_scaler.pkl', 'rb') )
        self.store_type_encoder            = pickle.load( open( self.home_path + 'parameter/store_type_encoder.pkl', 'rb') )

    def data_cleaning(self, df):
        
        # Rename columns
        df.columns = [inflection.underscore(col) for col in df.columns]

        ## Change Data Types
        df['date'] = pd.to_datetime(df['date'])

        ## Fillout NA
        # competition_distance -  suposição que quando competition_distance é NA significa que a loja não tem um competido próximo
        df['competition_distance'] = df['competition_distance'].apply(lambda x: 200000 if pd.isna(x) else x)
                
        # competition_open_since_month  
        df['competition_open_since_month'] = df[['competition_open_since_month', 'date']].apply(lambda x: x['date'].month if math.isnan(x['competition_open_since_month']) else x['competition_open_since_month'], axis=1)

        # competition_open_since_year 
        df['competition_open_since_year'] = df[['competition_open_since_year', 'date']].apply(lambda x: x['date'].year if math.isnan(x['competition_open_since_year']) else x['competition_open_since_year'], axis=1)

        # promo2_since_week 
        df['promo2_since_week'] = df[['promo2_since_week', 'date']].apply(lambda x: x['date'].week if math.isnan(x['promo2_since_week']) else x['promo2_since_week'], axis=1)            

        # promo2_since_year   
        df['promo2_since_year'] = df[['promo2_since_year', 'date']].apply(lambda x: x['date'].year if math.isnan(x['promo2_since_year']) else x['promo2_since_year'], axis=1)

        # promo_interval
        df['promo_interval'].fillna(0, inplace=True)

        month_map = {1 : 'Jan', 2 : 'Fev', 3 : 'Mar', 4 : 'Apr', 5 : 'May', 6 : 'Jun', 7 : 'Jul', 8 : 'Aug', 9 : 'Sep', 10 : 'Oct', 11 : 'Nov', 12 : 'Dec'}
        df['month_map'] = df['date'].dt.month.map(month_map)

        df['is_promo'] = df[['promo_interval', 'month_map']].apply(lambda x: 0 if x['promo_interval'] == 0 else
                                                                            1 if x['month_map'] in x['promo_interval'].split(',') else
                                                                            0, axis =1) 

        ## Change Data Types
        df['competition_open_since_month'] = df['competition_open_since_month'].astype(int)
        df['competition_open_since_year'] = df['competition_open_since_year'].astype(int)
        df['promo2_since_week'] = df['promo2_since_week'].astype(int)
        df['promo2_since_year'] = df['promo2_since_year'].astype(int)

        return df

    def feature_engineering( self, df ):
        
        # year
        df['year'] = df['date'].dt.year

        # month
        df['month'] = df['date'].dt.month

        # day
        df['day'] = df['date'].dt.day

        # week of year
        df['week_of_year'] = df['date'].dt.weekofyear

        # year week
        df['year_week'] = df['date'].dt.strftime('%Y-%W')

        # competition since
        df['competition_since'] = df.apply(lambda x: datetime.datetime(year=x['competition_open_since_year'], month=x['competition_open_since_month'], day=1), axis=1)
        df['competition_time_month'] = ((df['date'] - df['competition_since'])/np.timedelta64(1,'M'))

        # promo since
        df['promo_since'] = pd.to_datetime(df['promo2_since_year'].astype(str) + '-' + df['promo2_since_week'].astype(str) + '-1', 
                                            format='%Y-%W-%w') - pd.Timedelta(days=7)

        df['promo_time_week'] = ((df['date'] - df['promo_since'])/np.timedelta64(1,'W'))

        # assortment
        df['assortment'] = df['assortment'].apply(lambda x: 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended')

        # state_holiday
        df['state_holiday'] = df['state_holiday'].apply(lambda x: 'public_holiday' if x == 'a' else 'easter_holiday' if x == 'b' else 'christmas' if x == 'c' else 'regular_day')

        ## Feature Filtering
        df = df[(df['open'] != 0)]

        cols_drop = ['open', 'promo_interval', 'month_map']
        df = df.drop(cols_drop, axis=1)

        return df

    def data_preparation( self, df ):

        # competition_distance
        df['competition_distance'] = self.competition_distance_scaler.transform(df[['competition_distance']])

        # competition_time_month
        df['competition_time_month'] = self.competition_time_month_scaler.transform(df[['competition_time_month']])

        # promo_time_week
        df['promo_time_week'] = self.promo_time_week_scaler.transform(df[['promo_time_week']])

        # year
        df['year'] = self.year_scaler.transform(df[['year']])

        # state_holiday - One Hot Encoding
        df = pd.get_dummies(df, prefix=['state_holiday'], columns=['state_holiday'], drop_first=True)

        # store_type
        df['store_type'] = self.store_type_encoder.transform(df['store_type'])

        # assortment - Ordinal Encodin
        assotment_dict = {'basic': 1, 'extended': 2, 'extra': 3}
        df['assortment'] = df['assortment'].map(assotment_dict)

        ## Natute Transformation
        # month
        df['month_sin'] = df['month'].apply(lambda x: np.sin(x * (2 * np.pi / 12)))
        df['month_cos'] = df['month'].apply(lambda x: np.cos(x * (2 * np.pi / 12)))

        # day_of_week
        df['day_of_week_sin'] = df['day_of_week'].apply(lambda x: np.sin(x * (2 * np.pi / 7)))
        df['day_of_week_cos'] = df['day_of_week'].apply(lambda x: np.cos(x * (2 * np.pi / 7)))

        # day
        df['day_sin'] = df['day'].apply(lambda x: np.sin(x * (2 * np.pi / 31)))
        df['day_cos'] = df['day'].apply(lambda x: np.cos(x * (2 * np.pi / 31)))

        # week_of_year
        df['week_of_year_sin'] = df['week_of_year'].apply(lambda x: np.sin(x * (2 * np.pi / 52)))
        df['week_of_year_cos'] = df['week_of_year'].apply(lambda x: np.cos(x * (2 * np.pi / 52)))

        cols_selected = ['store', 'promo', 'store_type', 'assortment', 'competition_distance', 'competition_open_since_month', 'competition_open_since_year',
                         'promo2', 'promo2_since_week', 'promo2_since_year', 'competition_time_month', 'promo_time_week', 'month_cos',
                         'day_of_week_sin', 'day_of_week_cos', 'day_sin', 'day_cos']

        return df[ cols_selected ]
    
    def get_prediction(self, model, original_data, test_data ):
        # prediction
        pred = model.predict( test_data )
        
        # join pred into the original data
        original_data['prediction'] = np.expm1( pred )
        
        return original_data.to_json( orient='records', date_format='iso' )