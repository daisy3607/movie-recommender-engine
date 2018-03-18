import pandas as pd
import numpy as np
import keras
from keras.models import Sequential, Input, Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Embedding, Reshape, Merge, Dropout, Dense, Flatten, dot, BatchNormalization, add
import sys

#load data
#ratings = pd.read_csv('../data/train.csv',sep=',')
test_data = pd.read_csv(sys.argv[1])
test_userid = test_data['UserID'].values-1
test_movieid = test_data['MovieID'].values-1

#parameters
num_users = test_data['UserID'].drop_duplicates().max()
num_movies = test_data['MovieID'].drop_duplicates().max()

#load model

from keras.models import Sequential, Input, Model, load_model

class MFModel(Model):
    def __init__(self, num_users=num_users, num_movies=num_movies, K=120, **kwargs):        
        input_user = Input(shape=(1,))
        input_movie = Input(shape=(1,))
        
        user_layer = Embedding(num_users, K, input_length=1)(input_user)
        user_layer = Reshape((K,))(user_layer)
        user_layer = Dense(32)(user_layer)
        user_layer = keras.layers.PReLU()(user_layer)
        user_layer = BatchNormalization()(user_layer)
        user_layer = Dropout(0.5)(user_layer)
#   
        movie_layer = Embedding(num_movies, K, input_length=1)(input_movie)
        movie_layer = Reshape((K,))(movie_layer)
        movie_layer = Dense(32)(movie_layer)
        movie_layer = keras.layers.PReLU()(movie_layer)
        movie_layer = BatchNormalization()(movie_layer)
        movie_layer = Dropout(0.5)(movie_layer)
        
        user_bias = Embedding(num_users, K, input_length=1)(input_user)
        user_bias = Reshape((K,))(user_bias)
        user_bias = Dense(32)(user_bias)
        user_bias = keras.layers.PReLU()(user_bias)
        user_bias = BatchNormalization()(user_bias)
        user_bias = Dropout(0.5)(user_bias)
        user_bias = Dense(1)(user_bias)
        
        movie_bias = Embedding(num_movies, K, input_length=1)(input_movie)
        movie_bias = Reshape((K,))(movie_bias)
        movie_bias = Dense(32)(movie_bias)
        movie_bias = keras.layers.PReLU()(movie_bias)
        movie_bias = BatchNormalization()(movie_bias)
        movie_bias = Dropout(0.5)(movie_bias)
        movie_bias = Dense(1)(movie_bias)
        
        result = add([dot([user_layer, movie_layer], axes=1), user_bias, movie_bias])
        super(MFModel, self).__init__(inputs=[input_user, input_movie], outputs=result)

model = load_model('model/model855', custom_objects={ 'MFModel': MFModel})


#predict
result = model.predict([np.array(test_userid), np.array(test_movieid)])
predict_answer = result.flatten()
predict_answer = [1 if x < 1 else x for x in predict_answer]
predict_answer = [5 if x > 5 else x for x in predict_answer]

#submission
submission_data = pd.read_csv('data/SampleSubmisson.csv',sep=',',header=0)
out_df = submission_data.copy()
out_df['Rating'] = predict_answer
out_df.to_csv(sys.argv[2],index=None)


