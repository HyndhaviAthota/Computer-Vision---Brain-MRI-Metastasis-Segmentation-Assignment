from tensorflow.keras.preprocessing.image import ImageDataGenerator

def augment_training_data(image_set, mask_set):

    augmentation_params = dict(rotation_range=30,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                zoom_range=0.2,
                                horizontal_flip=True)
    
    image_generator = ImageDataGenerator(**augmentation_params)
    mask_generator = ImageDataGenerator(**augmentation_params)

    return image_generator, mask_generator
