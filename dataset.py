from keras.datasets import mnist
from keras.utils import  to_categorical

# load MNIST dataset...

def dataset():
      (x_train, y_train), (x_test, y_test) = mnist.load_data()
      image_size = x_train.shape[1]
      x_train = x_train.reshape((-1, image_size, image_size, 1)).astype('float32')/255
      y_train = to_categorical(y_train)
      return x_train, y_train, x_test, y_test
