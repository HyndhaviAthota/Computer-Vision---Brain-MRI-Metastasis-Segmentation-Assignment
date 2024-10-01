from models.nested_unet import nested_unet
from preprocess.clahe_preprocess import preprocess_images
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

dataset_path = 'dataset/' 
input_images, target_masks = preprocess_images(dataset_path)

unet_model = nested_unet()
unet_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

best_weights = ModelCheckpoint('weights/unet_best_weights.h5', monitor='val_loss', save_best_only=True)

unet_model.fit(train_inputs, train_targets, validation_data=(val_inputs, val_targets), epochs=50, callbacks=[best_weights])
