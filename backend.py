from sklearn.metrics import jaccard_score
from models.nested_unet import nested_unet

def evaluate_network(network, x_test, y_test):
    """Evaluate model performance using DICE Score."""
    output = network.predict(x_test)
    jaccard_index = jaccard_score(y_test.flatten(), output.flatten(), average='binary')
    return jaccard_index
