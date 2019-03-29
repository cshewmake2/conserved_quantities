import keras
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers


def hi():
	print("hi")

def train_encoder(state_derivs, encoded_dim, num_layers, l1_penalty):
		
	(num_data_pts, input_dim) = state_derivs.shape

	print(input_dim)
	input_layer = Input(shape=(input_dim,))

	encoded_layer = Dense(encoded_dim, activation = 'relu', activity_regularizer=regularizers.l1(l1_penalty))(input_layer)

	decoded_layer = Dense(input_dim, activation = 'linear')(encoded_layer)

	
	encoder = Model(input_layer, encoded_layer)

	autoencoder = Model(input_layer, decoded_layer)

	encoded_input = Input(shape=(encoded_dim,))
	# retrieve the last layer of the autoencoder model
	decoder_layer = autoencoder.layers[-1]
	# create the decoder model
	decoder = Model(encoded_input, decoder_layer(encoded_input))


	autoencoder.compile(optimizer = 'adadelta', loss = 'mean_squared_error')

	print(state_derivs)

	autoencoder.fit(state_derivs, state_derivs,
                epochs=500,
                batch_size=256,
                shuffle=True)

	return [autoencoder, encoder, decoder]