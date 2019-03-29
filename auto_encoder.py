import keras
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import Sequential
from keras import regularizers


def hi():
	print("hi")

# def train_encoder(state_derivs, encoded_dim, num_layers, l1_penalty):
		
# 	(num_data_pts, input_dim) = state_derivs.shape

# 	print(input_dim)
# 	input_layer = Input(shape=(input_dim,))

# 	left_layers = []

# 	for n in range(num_layers):
# 		if n == 0:
# 			encoded_layer_current = Dense(input_dim, activation = 'tanh')(input_layer)
# 			left_layers.append(encoded_layer_current)
# 		else:
# 			encoded_layer_current = Dense(input_dim, activation = 'tanh')(left_layers[n-1])
# 			left_layers.append(encoded_layer_current)


# 	encoded_layer = []
# 	if num_layers == 0:
# 		encoded_layer = Dense(encoded_dim, activation = 'tanh', activity_regularizer=regularizers.l1(l1_penalty))(input_layer)
# 	else:
# 		encoded_layer = Dense(encoded_dim, activation = 'tanh', activity_regularizer=regularizers.l1(l1_penalty))(left_layers[len(left_layers)-1])


# 	right_layers = []
# 	for n in range(num_layers):
# 		if n == 0:
# 			encoded_layer_current = Dense(input_dim, activation = 'tanh')(input_layer)
# 			right_layers.append(encoded_layer_current)
# 		else:
# 			encoded_layer_current = Dense(input_dim, activation = 'tanh')(right_layers[n-1])
# 			right_layers.append(encoded_layer_current)	


# 	decoded_layer = []
# 	if num_layers == 0:
# 		decoded_layer = Dense(input_dim, activation = 'linear')(encoded_layer)

# 	else:
# 		decoded_layer = Dense(input_dim, activation = 'linear')(right_layers[len(right_layers)-1])
	


# 	encoder = Model(input_layer, encoded_layer)

# 	autoencoder = Model(input_layer, decoded_layer)

# 	encoded_input = Input(shape=(encoded_dim,))
# 	# retrieve the last layer of the autoencoder model
# 	decoder_layer = autoencoder.layers[-1]
# 	# create the decoder model
	
# 	if n==0:
# 		decoder = Model(encoded_input, decoder_layer(encoded_input))
# 	else:
# 		decoder = Model(encoded_input, )

# 	autoencoder.compile(optimizer = 'adadelta', loss = 'mean_squared_error')

# 	print(state_derivs)

# 	autoencoder.fit(state_derivs, state_derivs,
#                 epochs=500,
#                 batch_size=256,
#                 shuffle=True)

# 	return [autoencoder, encoder, decoder]




def train_encoder(state_derivs, encoding_dim, num_layers, l1_penalty):
		
	(num_data_pts, input_dim) = state_derivs.shape

	print(input_dim)
	input_layer = Input(shape=(input_dim,))

	autoencoder = Sequential()

	for n in range(num_layers):
		if n==0:

			autoencoder.add( Dense(input_dim, input_shape=(input_dim,), activation='tanh') )
		else:
			autoencoder.add( Dense(input_dim, activation='tanh'))

	if num_layers == 0:
		autoencoder.add(    Dense(encoding_dim, input_shape=(input_dim,), activation='tanh') )
	else:
		autoencoder.add(  Dense(encoding_dim, activation='tanh') )

	for n in range(num_layers):
		if n==0:

			autoencoder.add( Dense(input_dim, input_shape=(input_dim,), activation='tanh') )
		else:
			autoencoder.add( Dense(input_dim, activation='tanh'))


	encoding_layer = autoencoder.layers[num_layers]
	encoder = Model(input)
	decoder = Sequential()


	autoencoder.add( Dense(input_dim, activation='linear') )

	autoencoder.compile(optimizer = 'adadelta', loss = 'mean_squared_error')

	print(state_derivs)

	autoencoder.fit(state_derivs, state_derivs,
                epochs=500,
                batch_size=256,
                shuffle=True)

	return [autoencoder, encoder, decoder]