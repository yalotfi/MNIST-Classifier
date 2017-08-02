from Classifier.preprocess import Preprocessor
from Classifier.ConvNet import ConvNetBuilder
from Classifier.ModelIO import ConvNetIO


# Process and build computational graph of ConvNet
pr = Preprocessor()
cnn = ConvNetBuilder(pr.input_shape, pr.num_classes)
model = cnn.compile_graph()

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
print('Accuracy: ', score[1])

# Save model in './Classifier/model_stores'
io = ConvNetIO(model)
io.save_model()
io.load_trained_model()
