from model.CNN import CNN
import numpy as np
from .model.FFNN.NN import NN

def main():
    cnn = CNN(layers=[[3, 8], [3, 16]], first_in_channels=1)
    dummy_img = np.zeros((28, 28))
    flat_len = cnn.feed_forward(dummy_img)
    print("flat =", len(flat_len))  #400
    nn = NN([flat_len, 30, 10])




if (__name__=="__main__"):
    main()