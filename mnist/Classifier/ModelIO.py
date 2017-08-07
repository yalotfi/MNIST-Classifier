from keras.models import model_from_json


def save_model(model, architecture_path, weight_path):
    # Save architecture in jSON file first
    model_json = model.to_json()
    with open(architecture_path, "w") as json_file:
        json_file.write(model_json)

    # Serialize model weights to HDF5
    model.save_weights(weight_path)
    print("Model architecture and weights saved to disk")


def load_trained_model(architecture_path, weight_path):
    # Load model architecture
    json_file = open(architecture_path, 'r')
    model_architecture = json_file.read()
    json_file.close()

    # Compile model and load trained weights
    model = model_from_json(model_architecture)
    model.load_weights(weight_path)
    print("Model architecture and weights loaded from disk")
