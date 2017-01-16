import numpy as np
import pandas as pd
import cv2,json,random
import matplotlib.image as mpimg
from skimage.exposure import rescale_intensity
from matplotlib.colors import rgb_to_hsv

from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model, Sequential
from keras.layers import Activation, MaxPooling2D, Flatten, Dropout
from keras.layers import Dense, GlobalAveragePooling2D, Input, Convolution2D
from keras.layers import Lambda, ELU, Merge, BatchNormalization
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json

# global variable of image size and batch size
input_height = 64
input_width = 64
batch_size = 128

def _random_image_path_and_angle(data_frame, index):
	"""
	Augmenting: random image path from center, left or right camera
	"""
	# default offset is 0.25
	OFF_CENTER_IMG = .25
	angle = data_frame.iloc[index].steering
	random_index = np.random.randint(2)

	if angle < 0:
		if random_index == 0:
			image_path = data_frame.iloc[index].right[1:]
			angle -= OFF_CENTER_IMG
		else:
			image_path = data_frame.iloc[index].center
	else:
		if random_index == 0:
			image_path = data_frame.iloc[index].left[1:]
			angle += OFF_CENTER_IMG
		else:
			image_path = data_frame.iloc[index].center

	return image_path, angle

def _crop(image):
	"""
	Preprocessing: crop
	"""
	h = image.shape[0]
	w = image.shape[1]
	cropped = image[int(h/5.):h-25,:,:]
	return cropped

def _resize(image, height, width):
	"""
	Preprocessing: resize
	"""
	resized = cv2.resize(image, (height, width), interpolation=cv2.INTER_AREA)
	return resized

def _random_angle_translate(angle):
	"""
	Augmenting: random angle translation
	"""
	TRANSLATE_X_RANGE = 100
	TRANSLATE_ANGLE = .2
	x_translation = TRANSLATE_X_RANGE*np.random.uniform()-TRANSLATE_X_RANGE/2
	translated_angle = angle + x_translation/TRANSLATE_X_RANGE*2*TRANSLATE_ANGLE
	return translated_angle, x_translation

def _random_image_translate(image, x_translation):
	"""
	Augmenting: random image translation
	"""
	TRANSLATE_Y_RANGE = 40
	y_translation = TRANSLATE_Y_RANGE*np.random.uniform()-TRANSLATE_Y_RANGE/2
	translation_matrix = np.float32([[1, 0, x_translation], [0, 1, y_translation]])
	translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
	return translated_image

def _random_brightness(image):
	"""
	Augmenting: random brightness
	"""
	BRIGHTNESS_RANGE = .25
	hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	random_brightness = BRIGHTNESS_RANGE + np.random.uniform()
	hsv_image[:,:,2] = hsv_image[:,:,2] * random_brightness
	rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
	return rgb_image

def _random_shadow(image):
	"""
	Augmenting: randomly located shadow
	"""
	top_y = 320*np.random.uniform()
	top_x = 0
	bot_x = 160
	bot_y = 320*np.random.uniform()
	image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
	shadow_mask = 0*image_hls[:,:,1]
	X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
	Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
	shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
	if np.random.randint(2)==1:
		random_bright = .5
		cond1 = shadow_mask==1
		cond0 = shadow_mask==0
		if np.random.randint(2)==1:
			image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
		else:
			image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright
	image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
	return image

def _random_vertical_flip(image, angle):
	"""
	Augmenting: random image flip
	"""
	flipped_image = np.fliplr(image)
	flipped_image_angle = -angle
	if np.random.randint(2) == 0:
		return flipped_image, flipped_image_angle
	else:
		return image, angle



def random_augment_image_and_angle(data_frame, resize_height, resize_width):
	"""
	Randomly augment and preprocess random image
	"""
	num_rows = data_frame.shape[0]
	index = np.random.randint(num_rows)
	image_path, angle = _random_image_path_and_angle(data_frame, index)
	angle, x_translation = _random_angle_translate(angle)
	image = load_img(image_path)
	image = img_to_array(image)
	image = _random_image_translate(image, x_translation)
	image = _random_brightness(image)
	image = _random_shadow(image)
	image, angle = _random_vertical_flip(image, angle)
	image = _crop(image)
	image = _resize(image, resize_height, resize_width)
	return image, angle

def data_generator(data_frame, resize_height, resize_width, batch_size=128):
	"""
	Train/val data generator
	"""
	X_batch = np.zeros((batch_size, resize_height, resize_width, 3), dtype=np.float)
	y_batch = np.zeros(batch_size, dtype=np.float)
	index = 0
	while 1:
		image, angle = random_augment_image_and_angle(data_frame, resize_height, resize_width)
		X_batch[index] = image
		y_batch[index] = angle
		index += 1

		if index >= batch_size:
			yield X_batch, y_batch
			X_batch = np.zeros((batch_size, resize_height, resize_width, 3), dtype=np.float)
			y_batch = np.zeros(batch_size, dtype=np.float)
			index = 0

def save_model_and_weights():
	"""
	Saving model and weights
	"""
	model.save_weights('model.h5')
	with open('model.json', 'w') as file:
		json.dump(model.to_json(), file)
	print("Model and weights saved")

def comma_ai_model(input_height, input_width):

	"""
	Comma model with additional normalization and dropout layers
	"""

	input_shape = (input_height, input_width, 3)

	model = Sequential()

	# Normalization
	model.add(Lambda(lambda x: x/255.-0.5,input_shape=input_shape))

	# Conv1
	model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", name='Conv1', activation='relu'))
	# Conv2
	model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same", name='Conv2', activation='relu'))
	# Conv3
	model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same", name='Conv3'))

	# Flatten Layer
	model.add(Flatten())
	model.add(Activation('relu'))
	model.add(Dense(1024, name='Dense1', init='he_normal', activation='relu'))
	model.add(Dropout(0.5, name='Dropout'))
	model.add(Dense(1, name='Output', init='he_normal', activation='linear'))

	return model

def train_model(model, data_frame_train, data_frame_val, epochs=20):
	"""
	Training model
	"""
	start_epoch = 0
	end_epoch = epochs
	batches_per_epoch = int(data_frame_train.shape[0]/batch_size)
	model.compile(optimizer="adam", loss="mse")

	# Evaluation Pre-train
	val_gen = data_generator(data_frame_val, input_height, input_width, batch_size=batch_size)
	val_loss = model.evaluate_generator(val_gen, val_samples=batch_size)
	print('Pre-train evaluation loss = {}'.format(val_loss))

	# Actual Batch Training
	epoch = start_epoch
	while True:
		print('Epoch {}/{}'.format(epoch + 1, end_epoch), end=': ')

		train_gen = data_generator(data_frame_train, input_height, input_width, batch_size=batch_size)
		val_gen = data_generator(data_frame_val, input_height, input_width, batch_size=batch_size)

		history = model.fit_generator(
			train_gen,
			samples_per_epoch = batches_per_epoch * batch_size,
			nb_epoch = 1,
			validation_data=val_gen,
			nb_val_samples = batch_size,
			verbose = 1)

		save_model_and_weights()

		epoch += 1
		if epoch >= end_epoch:
			break

if __name__ == '__main__':
	# Load the Udacity data
	data_frame = pd.read_csv("driving_log.csv")
	num_rows = data_frame.shape[0]

	# Split the data into training and validation set
	data_frame_val, data_frame_train = np.split(data_frame.sample(frac=1), [batch_size])

	# Create the final model
	model = comma_ai_model(input_height, input_width)
	model.summary()

	# Train the model with image generator
	train_model(model, data_frame_train, data_frame_val, epochs=20)

	exit(0)
