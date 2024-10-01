from sklearn.metrics import jaccard_score
from models.nested_unet import nested_unet

def assess_model(model_instance, images_test, masks_test):
    """Evaluate model performance using DICE Score."""
    predictions = model_instance.predict(images_test)
    dice_coefficient = jaccard_score(masks_test.flatten(), predictions.flatten(), average='binary')
    return dice_coefficient
