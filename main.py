from model.CNN import CNN
import numpy as np
from .model.FFNN.NN import NN
from utils.mnist_loader import load_data_wrapper


def main():
    cnn = CNN(layers=[[3, 8], [3, 16]], first_in_channels=1)
    dummy_img = np.zeros((28, 28))
    flat_len = cnn.feed_forward(dummy_img)
    print("flat =", len(flat_len))  #400
    nn = NN([flat_len, 30, 10])
    training_data , validation_data , test_data = load_data_wrapper()
    print("------------Data was loaded------------------")
    print(training_data)



if (__name__=="__main__"):
    main()