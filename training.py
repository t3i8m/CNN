
from model.CNN import CNN
from utils.mnist_loader import load_data_wrapper

cnn = CNN(layers=[[3, 8], [3, 16]], first_in_channels=1)
training_data, validation_data, test_data = load_data_wrapper()

cnn.SGD(training_data, 30, 32, 3.0, test_data = test_data)
cnn.save_weights("checkpoints/epoch30.npz")   