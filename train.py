import pandas as pd
import numpy as np
import keras
from keras.models import Sequential, Input, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Embedding, Reshape, Merge, Dropout, Dense, Flatten, dot, BatchNormalization, add

#load data
ratings = pd.read_csv('../data/train.csv',sep=',')
user_features = pd.read_csv('../data/users.csv',sep='::')
movie_features = pd.read_csv('../data/movies.csv',sep='::')

#shuffle
shuffled_ratings = ratings.sample(frac=1., random_state=1446557)
UserID = shuffled_ratings['UserID'].values-1
MovieID = shuffled_ratings['MovieID'].values-1
Ratings = shuffled_ratings['Rating'].values

#data
num_users = ratings['UserID'].drop_duplicates().max()
num_movies = ratings['MovieID'].drop_duplicates().max()


K=128       #latent dimension

class MFModel(Model):
    def __init__(self, num_users, num_movies, K, **kwargs):        
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
 
model = MFModel(num_users, num_movies, K)
model.compile(loss='mse', optimizer='adamax')

callbacks = [EarlyStopping('val_loss', patience=5), 
             ModelCheckpoint('../model/model_test', save_best_only=True)]
history = model.fit([UserID, MovieID], Ratings,batch_size=128, epochs=30, validation_split=.1, verbose=1, callbacks=callbacks)

# rmse = np.sqrt(history.history['val_loss'][-1])
# print(rmse)