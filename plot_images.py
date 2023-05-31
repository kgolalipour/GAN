import numpy as np
import matplotlib.pyplot as plt


def plot_images(generator, noise_input, labels, step, show=False):

    image_address = 'images'

    z = noise_input
    n_images = z.shape[0]

    rows = np.sqrt(n_images)
    plt.figure(figsize=(2, 2))
    images = generator.predict([z, labels])
    image_size = images.shape[1]

    for i in range(n_images):
        plt.subplot(rows ,rows, i+ 1)
        plt.imshow(images[i].reshape((image_size, image_size)), cmap='gray')
        plt.axis('off')
    plt.savefig(os.path.join(image_address, f"{step}.png"))

    if show:
        plt.show()
    else:
        plt.close('all')
