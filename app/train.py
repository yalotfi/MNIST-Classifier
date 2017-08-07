from Classifier.preprocess import Preprocessor
from Classifier.ConvNet import ConvNetBuilder
from Classifier.ModelIO import save_model


# Process and build computational graph of ConvNet
pr = Preprocessor()
pr.format_mnist()
print('\nx_train shape: ', pr.x_train.shape)
print(pr.x_train.shape[0], ' training examples')
print(pr.x_test.shape[0], ' test examples\n')

# Prepare the model
cnn = ConvNetBuilder(pr.input_shape, pr.num_classes)
model = cnn.compile_graph()
print('{}\n'.format(model.summary()))

# Hyperparameters for training
batch_size = 128
epochs = 12

# Train the ConvNet
model.fit(pr.x_train, pr.y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(pr.x_test, pr.y_test))
score = model.evaluate(pr.x_test, pr.y_test, verbose=0)
print('Error: ', score[0])
print('Accuracy: {}\n'.format(score[1]))

# Save model in './app/model'
architecture_path = 'model/architecture.json'
weight_path = 'model/weights.h5'
save_model(model, architecture_path, weight_path)
