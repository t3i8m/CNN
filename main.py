from model.CNN import CNN
import numpy as np
from model.FFNN.NN import NN
from utils.mnist_loader import load_data_wrapper
import matplotlib.pyplot as plt

def main():
    cnn = CNN(layers=[[3, 8], [3, 16]], first_in_channels=1)
    dummy_img = np.zeros((28, 28))
    flat_len = len(cnn.feed_forward(dummy_img))
    print("flat =", flat_len)  #400
    training_data , validation_data , test_data = load_data_wrapper()
    
    print("------------Data was loaded------------------")
    print(test_data[0])

    image, label = test_data[0]
    plt.imshow(image, cmap='gray')  
    plt.title(f"Label: {label}")
    plt.colorbar()
    plt.show()



if (__name__=="__main__"):
    main()