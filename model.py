import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv2D, LeakyReLU, Conv2DTranspose, Flatten
from keras.layers import Layer, Reshape, BatchNormalization, Activation, Input, concatenate
from keras.models import Model
from keras.datasets import mnist
from keras.optimizers import RMSprop
from keras.utils import plot_model, to_categorical


def model(x_train, y_train, x_test, y_test, latent_dim,image_size ,image_shape , gen_filters,gen_strides ,dis_filters , dis_strides,kernel_size,alpha,batch_size,dis_lr,dis_decay,dis_optimizer,adv_lr,adv_decay,adv_optimizer,save_intervals,log_print_steps,test_size):
    train(models)
    build_and_train()
    return generator,noise_input,labels


def bn_relu(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


# we have 2 loss

def build_generator(z_inputs, label_inputs, image_size=28):
    filters = gen_filters

    x = concatenate([z_inputs, label_inputs])

    image_resize = image_size // 4  # 2 number of Conv2DTranspose (7 in Dense layer) (7*7*28)

    x = Dense(image_resize * image_resize * filters[0])(x)  # in model
    x = Reshape((image_resize, image_resize, filters[0]))(x)  # in model

    for strides, filter in zip(gen_strides, filters):
        x = bn_relu(x)
        x = Conv2DTranspose(filters=filter,
                            kernel_size=kernel_size,
                            padding='same',
                            strides=strides
                            )(x)
    outputs = Activation('sigmoid', name='Sigmoid')(x)
    model = Model([z_inputs, label_inputs], outputs, name='generator')
    model.summary()
    plot_model(model, to_file='generator.png', show_shapes=True)
    return model


def build_discriminator(inputs):
    x = inputs
    for strides, filter in zip(dis_strides, dis_filters):
        x = LeakyReLU(alpha=alpha)(x)
        x = Conv2D(filters=filter,
                   kernel_size=kernel_size,
                   padding='same',
                   strides=strides
                   )(x)
    y = Flatten()(x)

    x = Dense(1)(y)  # in model
    rf_outputs = Activation('sigmoid', name='Sigmoid')(x)

    # other path in model : (for classification)

    c = Dense(128)(y)
    c = Dense(10)(c)
    label_outputs = Activation('softmax', name='softmax')(c)

    # model has 2 outputs : (rf_outputs, label_outputs)
    model = Model(inputs, [rf_outputs, label_outputs], name='discriminator')
    return model


def build_and_train():  # compile models

    # first create discriminator (train discriminator) (compile)

    dis_inputs = Input(shape=image_shape, name='dis_inputs')
    dis = build_discriminator(dis_inputs)
    dis.compile(loss=['binary_crossentropy', 'categorical_crossentropy'],  # 2 loss ,output = 0,1
                optimizer=dis_optimizer,
                metrics=['acc']
                )

    # first create generator (train generator) (compile)

    gen_z_inputs = Input(shape=(latent_dim,), name='gen_z_inputs')  # for main input
    gen_label_inputs = Input(shape=(10,), name='gen_label_inputs')  # for label
    gen = build_generator(gen_z_inputs, gen_label_inputs)

    # create and compile a new overall model (adversarial model) : we update generator and discriminator simultaneously
    # since we don't use generetor alone

    dis.trainable = False  # when Adversarial is training, Discriminator must freeze
    adv_inputs = [gen_z_inputs, gen_label_inputs]  # adversarial input = generator input
    adv_outputs = dis(
        gen(adv_inputs))  # in Equation (paper). adversarial output = dicriminator output that input of discriminator in generator output that input of generator is adv_inputs.
    adv = Model(adv_inputs, adv_outputs, name='adversarial')
    adv.compile(loss=['binary_crossentropy', 'categorical_crossentropy'],
                optimizer=adv_optimizer,
                metrics=['acc']
                )
    adv.summary()
    plot_model(adv, to_file='adversarial.png', show_shapes=True)

    models = gen, dis, adv

    train(models)


def train(models):  # train models (Train the Discriminator and Adversarial Networks)

    gen, dis, adv = models
    m_train = x_train.shape[0]  # 60000

    # test_z is Fixed
    test_z = np.random.uniform(low=-1, high=+1, size=[test_size, latent_dim])
    test_lables = np.eye(10)[np.arange(test_size) % 10]

    for step in range(1, train_steps + 1):
        random_indices = np.random.randint(0, m_train, size=batch_size)  # select 64 elements, from 0 to 60000
        real_images = x_train[random_indices]  # select 64 real images

        z = np.random.uniform(low=-1, high=+1, size=[batch_size,
                                                     latent_dim])  # create noise, 64 number with size = latent_dim(100) with value -1,0,1
        fake_one_hot_labels = np.eye(10)[
            np.arange(batch_size) % 10]  # (new format showing) create fake one-hot labels. size = 64*10
        fake_images = gen.predict([z, fake_one_hot_labels])  # create fake images with z and labels

        fake_rf = np.zeros((batch_size, 1))  # binary classification is one column
        real_rf = np.ones((batch_size, 1))

        real_one_hot_labels = y_train[random_indices]

        dis_x = np.concatenate([real_images, fake_images])
        dis_y_rf = np.concatenate([real_rf, fake_rf])
        dis_y_labels = np.concatenate([real_one_hot_labels, fake_one_hot_labels])

        l, l_rf, acc_rf, l_labels, acc_labels = dis.train_on_batch(dis_x, [dis_y_rf, dis_y_labels])  # get one batch
        # l = sum 2 loss , l_rf = loss for r_f , acc_rf = accuracy for r_f , l_labels = loss for labels , acc_labels = accuracy for labels

        log = f'step:{step} dis[loss:{l}]'

        adv_z = np.random.uniform(low=-1, high=+1, size=[batch_size, latent_dim])
        adv_label = np.eye(10)[np.arange(batch_size) % 10]
        adv_x = [adv_z, adv_label]
        adv_y = [np.ones((batch_size, 1)), adv_label]  # from discriminator outputs, we expect labels=1

        l, l_rf, acc_rf, l_labels, acc_labels = adv.train_on_batch(adv_x, adv_y)
        log += f'adv[loss:{l}]'

        print(log)
        if step % save_intervals == 0:
            plot_images(gen, test_z, test_lables, step)


build_and_train()


def test_generator(generator):
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
    plot_images(generator,
                noise_input=noise_input,
                show=True,
                model_name="test_outputs")
