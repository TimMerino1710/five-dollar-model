import numpy as np
import os
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Conv2DTranspose, Conv2D, Add, ReLU, Layer, GlobalAveragePooling1D
from keras.layers import BatchNormalization, Activation, ZeroPadding3D, UpSampling2D, LeakyReLU, ZeroPadding2D, Attention, Permute, Multiply
from keras.activations import sigmoid, softmax
# from keras.layers.activation import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam
from keras.utils import plot_model

import matplotlib.pyplot as plt
import textwrap
from PIL import Image
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import csv
import json

seed = 32
np.random.seed(seed)
    

class ConditionalInstanceNormalization(Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super(ConditionalInstanceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, inputs):
        conv, scale, bias = inputs
        mean, variance = tf.nn.moments(conv, axes=[1, 2], keepdims=True)
        normalized_output = (conv - mean) / tf.sqrt(variance + self.epsilon)
        normalized_output *= scale[:, tf.newaxis, tf.newaxis]
        normalized_output += bias[:, tf.newaxis, tf.newaxis]
        return normalized_output
    

    
class FiLMLayer(Layer):
    def __init__(self):
        super(FiLMLayer, self).__init__()

    def call(self, inputs):
        # takes a list of inputs: [feature_map, film_params]
        ft, film_params = inputs

        # height, width, and number of features are taken from the feature_map. For conv layers, number of features = number of filters
        # have to do this or it errors when trying to save the model
        runtime_shape = tf.shape(ft)
        height, width, feature_size = runtime_shape[1], runtime_shape[2], runtime_shape[3]
        
        
        # for each film param, expands it to be the same dimension as the filter
        film_params = tf.expand_dims(film_params, axis=[1])
        film_params = tf.expand_dims(film_params, axis=[1])
        film_params = tf.tile(film_params, [1, height, width, 1])

        # first half of the film params is beta, second half is gamma
        gammas = film_params[:, :, :, :feature_size]
        betas = film_params[:, :, :, feature_size:]

        # computes the affine mapping in terms of gamma and beta, and returns
        output = (1 + gammas) * ft + betas

        return output
    

class DollarModel():
    def __init__(self, model_name, img_shape, lr, data_path, dataset_type, embedding_dim=128, z_dim=5, kern_size=7, filter_count=128, num_res_blocks=3, condition_type=''):

        self.init = tf.keras.initializers.HeNormal(seed=32)
        self.embedding_dim = embedding_dim
        self.img_shape = img_shape
        self.model_name = model_name
        self.test_set_size = 16
        self.z_dim = z_dim
        self.data_path = data_path
        self.filter_count = filter_count
        self.kern_size = kern_size
        self.dataset_type = dataset_type
        self.num_res_blocks = num_res_blocks
        self.condition_type = condition_type

        self.model_path = 'dollarmodel/' + self.model_name + "/models/"
        os.makedirs(self.model_path, exist_ok=True)
        self.sample_path = 'dollarmodel/' + self.model_name + "/samples/"
        os.makedirs(self.sample_path, exist_ok=True)
        self.tiles = self.extract_tiles('../map_tileset/pokemon_tileset.png')

        self.img_x = img_shape[0]
        self.img_y = img_shape[1]
        self.channels = img_shape[-1]
        self.img_shape = img_shape

        self.lr = lr

        self.load_data(scaling_factor=6)
        self.load_unseen(scaling_factor=6)


        if dataset_type == 'map':
            self.num_upsample = 2
            self.gen_to_image = self.map_to_image
        elif dataset_type == 'emoji':
            self.num_upsample = 2
            self.gen_to_image = self.sprite_to_image
        elif dataset_type == 'sprite':
            self.num_upsample = 1
            self.gen_to_image = self.sprite_to_image

        self.create_model()
        print(len(self.embeddings))

    def create_model(self):
        embedding = Input(shape=(self.embedding_dim,))
        noise = Input(shape=(self.z_dim,))

        enc_in_concat = Concatenate()([noise, embedding])
        x = Dense(self.filter_count * 4 * 4, kernel_initializer=self.init)(enc_in_concat)

        x = Reshape((4, 4, self.filter_count))(x)

        # add an upsampling layer, followed by a res block, followed by a conditioning layer
        for i in range(self.num_res_blocks):
            if i < self.num_upsample:
                x = UpSampling2D()(x)
            
            x1 = Conv2D(self.filter_count, kernel_size=self.kern_size, strides=1, padding="same", activation="relu", kernel_initializer=self.init)(x)
            x1 = BatchNormalization()(x1)
            x1 = Conv2D(self.filter_count, kernel_size=self.kern_size, strides=1, padding="same", activation="relu", kernel_initializer=self.init)(x1)
            x1 = BatchNormalization()(x1)
            x = Add()([x, x1])

            # add conditional instance normalization layer
            if self.condition_type == 'CIN':
                scale = Dense(self.filter_count, activation=None, name="CIN_scale" + str(i))(embedding)
                bias = Dense(self.filter_count, activation=None, name="CIN_bias" + str(i))(embedding)
                x = ConditionalInstanceNormalization()([x, scale, bias])
            # add FiLM layer
            elif self.condition_type == 'FiLM':
                film_param_gen = Dense(2 * self.filter_count, activation=None, name="film_param_gen_dense" + str(i))(embedding)
                x = FiLMLayer()([x, film_param_gen])
        
        if self.dataset_type == 'map':
            # output
            x = ZeroPadding2D(padding=(1, 1))(x)
            img = Conv2D(self.channels, kernel_size=9, padding="valid", activation="softmax", kernel_initializer=self.init)(x)
        else:
            img = Conv2D(self.channels, kernel_size=3, padding="same", activation="softmax", kernel_initializer=self.init)(x)

        self.generator = Model([noise, embedding], img, name="Generator")
        self.generator.compile(loss='categorical_crossentropy', optimizer=Adam(self.lr), metrics=['accuracy'])
        self.generator.summary()
        plot_model(self.generator, to_file=os.path.join(self.model_path, 'gen_model_graph.png'), show_shapes=True)



    
    def load_data(self, scaling_factor=6):
        path = self.data_path
        data = np.load(path, allow_pickle=True).item()
        self.images = np.array(data['images'])
        self.labels = data['labels']
        self.embeddings = data['embeddings']
        if isinstance(self.embeddings, list):
            self.embeddings = np.array(self.embeddings)
        self.embeddings = self.embeddings * scaling_factor

        if self.dataset_type == 'emoji' or self.dataset_type == 'sprite':
            self.color_palette = data['color_palette']
            self.color_palette_rgb = [tuple(int(color[i:i + 2], 16) for i in (1, 3, 5)) for color in self.color_palette]
            
        self.images, self.images_test, self.labels, self.labels_test, self.embeddings, self.embeddings_test = train_test_split(
        self.images, self.labels, self.embeddings, test_size=24, random_state=seed)

        print("Train size: ", len(self.labels))
        print("Test size: ", len(self.labels_test))


    # export the model to a path
    def exportModel(self, path):
        self.generator.save(path)

    def render_images(self, images, labels, title, save_path, embeddings=None, correct_images=None):
        num_images = len(images)
        num_rows = (num_images // 8) + ((num_images % 8) > 0)
        num_subplots_per_image = 1 + (embeddings is not None) + (correct_images is not None)
        
        # add an extra row for spacing after every 8 images
        fig, axs = plt.subplots((num_rows + num_rows // 8) * num_subplots_per_image, 8, figsize=(8*4, (num_rows + num_rows // 8)*4*num_subplots_per_image))

        # If there is only one row, axs will be a 1-dimensional array.
        if isinstance(axs, plt.Axes):
            axs = np.array([[axs]])

        for i, (image, label) in enumerate(zip(images, labels)):
            row = ((i // 8) + (i // 64)) * num_subplots_per_image
            col = i % 8

            pil_image = self.gen_to_image(image)  # assuming gen_to_image is defined elsewhere
            axs[row, col].imshow(pil_image)
            axs[row, col].set_title("\n".join(textwrap.wrap("GEN: " + label, width=30)), fontsize=8)
            axs[row, col].axis('off')

            if correct_images is not None:
                correct_pil_image = self.gen_to_image(correct_images[i])
                axs[row + 1, col].imshow(correct_pil_image)
                axs[row + 1, col].set_title("\n".join(textwrap.wrap("ORIG: " + label, width=30)), fontsize=8)
                axs[row + 1, col].axis('off')

            if embeddings is not None:
                embed_heatmap = axs[row + num_subplots_per_image - 1, col].imshow(np.reshape(embeddings[i], (16, 24)), cmap='hot', interpolation='nearest', vmin=-1, vmax=1)
                fig.colorbar(embed_heatmap, ax=axs[row + num_subplots_per_image - 1, col])
                axs[row + num_subplots_per_image - 1, col].axis('off')

        plt.suptitle(title, fontsize=16)
        plt.subplots_adjust(hspace=0.5)  # adjust space between rows
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

    def extract_tiles(self, image_file, tile_size=8, num_tiles=16):
        image = Image.open(image_file)
        tiles = []
        rows = cols = int(np.sqrt(num_tiles))

        for i in range(rows):
            for j in range(cols):
                tile = image.crop((j * tile_size, i * tile_size, (j + 1) * tile_size, (i + 1) * tile_size))
                tiles.append(tile)
        return tiles
    
    # Use tiles to construct image of map
    def map_to_image(self, ascii_map, tile_size=8):
        tiles = self.tiles
        rows, cols = ascii_map.shape
        image = Image.new('RGB', (cols * tile_size, rows * tile_size))

        for i in range(rows):
            for j in range(cols):
                tile_index = ascii_map[i, j]
                tile = tiles[tile_index]
                image.paste(tile, (j * tile_size, i * tile_size))

        return image

    def sprite_to_image(self, image):
        # Create an empty array with shape of the image and 3 color channels
        decoded_image = np.zeros((*image.shape, 3), dtype=np.uint8)

        for i in range(len(self.color_palette_rgb)):
            decoded_image[image == i] = self.color_palette_rgb[i]

        pil_image = Image.fromarray(decoded_image)
        return pil_image


######################################################################
#
#                               evals
#
######################################################################

# render samples from held out test set
def test_set_gen(ep, model, epoch_dir):
    title = model.model_name + ' Test set samples, epoch' + str(ep)
    file_name = os.path.join(epoch_dir, 'test_set_samples.png')

    embeddings = model.embeddings_test
    labels = model.labels_test
    correct_images = model.images_test

    noise = np.random.normal(0, 1, (len(embeddings), model.z_dim))
    predictions = model.generator.predict([noise, embeddings], verbose=False)

    argmaxed_gens = np.argmax(predictions, axis=-1)

    model.render_images(images=argmaxed_gens, labels=labels, correct_images=np.argmax(correct_images, axis=-1), title=title, save_path=file_name)


# render a single image with noise applied to the embedding
def emb_noise_gen(ep, model, image, label, embedding, noise_mu, noise_sd, epoch_dir, operation='add', num_gens=8):
    file_name = os.path.join(epoch_dir, 'noisy_embs_samples.png')
    if operation == 'add':
        title = model.model_name + 'embedding + N(' + str(noise_mu) + ", " + str(noise_sd ) + '), epoch' + str(ep)
        noisy_embs = [embedding + np.random.normal(noise_mu, noise_sd, model.embedding_dim) for _ in range(num_gens - 1)]
    else:
        title = model.model_name + 'embedding * N(' + str(noise_mu) + ", " + str(noise_sd ) + '), epoch' + str(ep)
        noisy_embs = [embedding * np.random.normal(noise_mu, noise_sd, model.embedding_dim) for _ in range(num_gens - 1)]

    noisy_embs.insert(0, embedding)
    noisy_embs = np.array(noisy_embs)

    labels = [label] * num_gens
    images = [image] * num_gens

    noise = np.random.normal(0, 1, (num_gens, model.z_dim))
    predictions = model.generator.predict([noisy_embs, noise], verbose=False)

    
    argmaxed_gens = np.argmax(predictions, axis=-1)
    model.render_images(images=argmaxed_gens, labels=labels, title=title, correct_images=np.argmax(images, axis=-1), save_path=file_name)


# render a single image with noise applied to the noise vector
def z_noise_gen(ep, model, image, label, embedding, noise_mu, noise_sd, epoch_dir, operation='add', num_gens=8, name=''):
    if name == '':
        file_name = os.path.join(epoch_dir, 'noisy_z_vec_samples.png')
    else:
        file_name = os.path.join(epoch_dir, name)
    z_vec = np.random.normal(0, 1, model.z_dim)

    if operation == 'add':
        title = model.model_name + 'noise vec + N(' + str(noise_mu) + ", " + str(noise_sd ) + '), epoch' + str(ep)
        noisy_z_vecs = [z_vec + np.random.normal(noise_mu, noise_sd, model.z_dim) for _ in range(num_gens - 1)]
    else:
        title = model.model_name + 'noise vec * N(' + str(noise_mu) + ", " + str(noise_sd ) + '), epoch' + str(ep)
        noisy_z_vecs = [z_vec * np.random.normal(noise_mu, noise_sd, model.z_dim) for _ in range(num_gens - 1)]

    noisy_z_vecs.insert(0, z_vec)
    noisy_z_vecs = np.array(noisy_z_vecs)

    embedddings = np.array([embedding] * num_gens)
    labels = [label] * num_gens
    images = [image] * num_gens

    predictions = model.generator.predict([embedddings, noisy_z_vecs], verbose=False)
    argmaxed_gens = np.argmax(predictions, axis=-1)

    model.render_images(images=argmaxed_gens, labels=labels, title=title, correct_images=np.argmax(images, axis=-1), save_path=file_name)
    

def do_renders(model, ep):
    epoch_dir = os.path.join(model.sample_path, 'epoch_' + str(ep))
    os.makedirs(epoch_dir, exist_ok=True)
    test_set_gen(ep, model, epoch_dir)


def train(model, epochs, batch_size, sample_interval=50):
    ep = 0
    loss = []
    acc = []
    val_loss = []
    val_acc = []

    do_renders(model, 0)

    for i in range(epochs // sample_interval):
        for i in range(sample_interval):
            noise = np.random.normal(0, 1, (len(model.embeddings), model.z_dim))
            hist = model.generator.fit(x=[noise, model.word_embeddings], y=model.images, validation_split=0.2, batch_size=batch_size, initial_epoch=ep + i, epochs=ep + i + 1, shuffle=True, verbose=2)
            loss = loss + hist.history['loss']
            acc = acc + hist.history['accuracy']
            val_loss = val_loss + hist.history['val_loss']
            val_acc = val_acc + hist.history['val_accuracy']
        ep += sample_interval
       
        do_renders(model, ep)
        model.exportModel(os.path.join(model.model_path, "generator_epoch" + str(ep)))

    done = sample_interval * (epochs // sample_interval)
    remaining = epochs - done
    if remaining > 0:
        for i in range (remaining):
            noise = np.random.normal(0, 1, (len(model.embeddings), model.z_dim))
            # noise = model.z_vectors
            hist = model.generator.fit(x=[noise, model.word_embeddings], y=model.images, validation_split=0.2, batch_size=batch_size, initial_epoch=ep + i, epochs=ep + i + 1, shuffle=True, verbose=2)
            loss = loss + hist.history['loss']
            acc = acc + hist.history['accuracy']
            val_loss = val_loss + hist.history['val_loss']
            val_acc = val_acc + hist.history['val_accuracy']
            total_loss = total_loss + hist.history['total loss']

    # if ep != epochs:
    ep = epochs
    do_renders(model, ep)
    model.exportModel(os.path.join(model.model_path, "generator_final"))


    # Find the index (epoch) of the lowest validation loss and the highest validation accuracy
    lowest_val_loss_epoch = np.argmin(val_loss)
    highest_val_acc_epoch = np.argmax(val_acc)

    # Plot the metrics
    plt.plot(loss, label='loss')
    plt.plot(acc, label='accuracy')
    plt.plot(val_loss, label='val_loss')
    plt.plot(val_acc, label='val_accuracy')

    # Add markers for the lowest validation loss and the highest validation accuracy
    plt.plot(lowest_val_loss_epoch, val_loss[lowest_val_loss_epoch], marker='o', markersize=8, label="Lowest val_loss: " + str(round(val_loss[lowest_val_loss_epoch], 3)) + ", ep " + str(lowest_val_loss_epoch), linestyle='None', color='red')
    plt.plot(highest_val_acc_epoch, val_acc[highest_val_acc_epoch], marker='o', markersize=8, label="Highest val_accuracy: " + str(round(val_acc[highest_val_acc_epoch], 3)) + ", ep " + str(highest_val_acc_epoch), linestyle='None', color='green')

    # Annotate the points with the epoch numbers
    plt.annotate(f"Epoch {lowest_val_loss_epoch}", (lowest_val_loss_epoch, val_loss[lowest_val_loss_epoch]), textcoords="offset points", xytext=(-10,7), ha='center', fontsize=8, color='red')
    plt.annotate(f"Epoch {highest_val_acc_epoch}", (highest_val_acc_epoch, val_acc[highest_val_acc_epoch]), textcoords="offset points", xytext=(-10,-15), ha='center', fontsize=8, color='green')

    # Add the legend and save the plot
    plt.legend()
    plt.suptitle(model.model_name, fontsize=16)
    plt.savefig(model.sample_path + "loss_graph.png")
    plt.close()

    return loss, acc, val_loss, val_acc, val_loss[lowest_val_loss_epoch], val_acc[highest_val_acc_epoch]

input_shape = (10, 10, 16)
epochs = 100
batch_size = 256
encpic = DollarModel(model_name="test_modelname", img_shape=input_shape, lr=0.0005, embedding_dim=384, z_dim=5, filter_count=128, kern_size=5, num_res_blocks=3, dataset_type='map', data_path='map_dataset.npy')
train(encpic, epochs, batch_size, sample_interval=10)





    
    
