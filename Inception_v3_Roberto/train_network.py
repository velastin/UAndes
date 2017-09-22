from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as K
import data_feed
import numpy as np

batch_size = 32

train_feed = data_feed.feed_data('train', samples=-1, batch_size=batch_size)
test_feed = data_feed.feed_data('test', samples=10000, batch_size=batch_size)

input_tensor = Input(shape=(224, 224, 3))  # this assumes K.image_data_format() == 'channels_last'

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- we have 2 classes
predictions = Dense(2, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# keras callbacks
checkpoint = ModelCheckpoint('D:\\PAMELA-UANDES\\Inception_V3\\best_model.hdf5', monitor='val_loss', save_best_only=True, save_weights_only=False)
tensorboard = TensorBoard(log_dir='D:\\PAMELA-UANDES\\Inception_V3\\tensorboard', write_graph=False)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

# train the model on the new data for a few epochs
log_factor = 1
warm_epochs = 2*log_factor
after_warm_epochs = 10*log_factor
model.fit_generator(
		generator = train_feed.generate(),
		steps_per_epoch = train_feed.length/(batch_size*log_factor),
		epochs = warm_epochs,
		validation_data = test_feed.generate(),
		validation_steps = test_feed.length/(batch_size*log_factor),
		initial_epoch = 0,
		callbacks=[checkpoint, tensorboard]
	)

print("Warm up finished. Now training *some* inception layers as well")

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
#for i, layer in enumerate(base_model.layers):
#   print(i, layer.name)

# we choose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import Adam
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['acc'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit_generator(
		generator = train_feed.generate(),
		steps_per_epoch = train_feed.length/(batch_size*log_factor),
		epochs = after_warm_epochs+warm_epochs,
		validation_data = test_feed.generate(),
		validation_steps = test_feed.length/(batch_size*log_factor),
		initial_epoch = warm_epochs,
		callbacks=[checkpoint, tensorboard]
	)

print("Model finished training. Saved best performing model to {}".format("D:\\PAMELA-UANDES\\Inception_V3\\best_model.hdf5"))
