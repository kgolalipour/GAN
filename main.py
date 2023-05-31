import numpy as np

from dataset import *
from plot_images import *
from model import *


# hyper parameters

latent_dim = 100   # dimention of random z(latent)
image_size = 28
image_shape = (image_size, image_size, 1)
# Generator
gen_filters = [128, 64, 32, 1]
gen_strides = [2, 2, 1, 1]
# Discrimunator
dis_filters = [32, 64, 128, 256]
dis_strides = [2, 2, 2, 1]
kernel_size = 5
strides = 2
alpha = 0.2   # parametr of LeakyReLU
batch_size = 64
dis_lr = 2e-4      # stable in paper
dis_decay = 6e-8   # stable in paper
dis_optimizer = RMSprop(lr=dis_lr, decay=dis_decay)
# Adversarial contains generator outputs get discriminator to binary classification
adv_lr = dis_lr * 0.5          # stable in paper
adv_decay = dis_decay * 0.5    # stable in paper
adv_optimizer = RMSprop(lr=adv_lr, decay=adv_decay)
save_intervals = 500
train_steps = 40000
log_print_steps = 50
test_size = 16


if __name__ == '__main__':
    x_train, y_train, x_test, y_test =  dataset()
    generator,noise_input,labels = model(x_train, y_train, x_test, y_test, latent_dim,image_size ,image_shape , gen_filters,gen_strides ,dis_filters , dis_strides,kernel_size,alpha,batch_size,dis_lr,dis_decay,dis_optimizer,adv_lr,adv_decay,adv_optimizer,save_intervals,log_print_steps,test_size)
    plot_images(generator, noise_input, labels, train_steps, show=False)

