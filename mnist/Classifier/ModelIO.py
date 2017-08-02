from keras.models import model_from_json


class ConvNetIO():
    """
    ModelIO is a set of input and output methods for trained Keras models.
    It inherits the model object as it relies on Keras primitives.
    """

    def __init__(self, model):
        super(ConvNetIO, self).__init__()
        self.model = model
        self.architecture_path = './model_stores/architecture.json'
        self.weight_path = './model_stores/weights.h5'

    def save_model(self):
        # Save architecture in jSON file first
        model_json = self.model.to_json()
        with open(self.architecture_path, "w") as json_file:
            json_file.write(model_json)

        # Serialize model weights to HDF5
        self.model.save_weights(self.weight_path)
        print("Model architecture and weights saved to disk")

    def load_trained_model(self):
        json_file = open(self.architecture_path, 'r')
        model_architecture = json_file.read()
        json_file.close()
        self.model = model_from_json(model_architecture)
        self.model.load_weights(self.weight_path)
        print("Model architecture and weights loaded from disk")
