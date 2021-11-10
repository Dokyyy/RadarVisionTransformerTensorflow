import tensorflow as tf
import numpy as np
from configuration import *
from convGen import *
from convCritic import *
from dataloader import *
import time

generator = ConvGen(
    image_size=256,
    patch_size=32,
    num_classes=1000,
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
)

discriminator = ConvCritic(
    image_size=256,
    patch_size=32,
    num_classes=1000,
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
)

# if torch.cuda.device_count() > 1 and multigpu:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#     model = nn.DataParallel(model)

# model.to(device)

# # check keras-like model summary using torchsummary
# from torchsummary import summary
# summary(model, input_size=(1, 256, 256))

pixel_mse = tf.keras.losses.MeanSquaredError()
pixel_opt = tf.keras.optimizers.Adam(learning_rate=.0001)

generator_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=.0001)
discriminator_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=.0001)


pix_loss_metric = tf.keras.metrics.Mean(name='pixel_loss')
gen_loss_metric = tf.keras.metrics.BinaryCrossentropy(from_logits=True, name='generator_loss')
dis_loss_metric = tf.keras.metrics.BinaryCrossentropy(from_logits=True, name='discriminator_loss')


@tf.function
def conv_gen_pixel_train_step(inputs, output):
    with tf.GradientTape() as tape:
        prediction = generator(inputs)
        pix_loss = pixel_mse(output, prediction)

    gradients = tape.gradient(pix_loss, generator.trainable_variables)
    pixel_opt.apply_gradients(zip(gradients, generator.trainable_variables))

    pix_loss_metric(pix_loss)
    return prediction


@tf.function
def conv_generator_train_step(inputs, trg_img=None):
    with tf.GradientTape() as tape:
        prediction = generator(inputs)
        dis_input = tf.keras.layers.concatenate((inputs, prediction), axis=-1)
        pred_class = discriminator(dis_input)
        gen_loss = generator_loss(tf.ones_like(pred_class), pred_class)

    gradients = tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))

    gen_loss_metric(tf.ones_like(pred_class), pred_class)
    return prediction


@tf.function
def conv_discriminator_train_step(inputs, trg_img):
    with tf.GradientTape() as tape:
        prediction = generator(inputs)
        dis_input = tf.keras.layers.concatenate((inputs, prediction), axis=-1)
        pred_class = discriminator(dis_input)
        dis_loss = discriminator_loss(tf.zeros_like(pred_class), pred_class)

    gradients = tape.gradient(dis_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))

    dis_loss_metric(tf.zeros_like(pred_class), pred_class)

    with tf.GradientTape() as tape:
        dis_input_1 = tf.keras.layers.concatenate((inputs, trg_img), axis=-1)
        pred_class_1 = discriminator(dis_input_1)
        dis_loss_1 = discriminator_loss(tf.ones_like(pred_class_1), pred_class_1)

    gradients = tape.gradient(dis_loss_1, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))

    dis_loss_metric(tf.ones_like(pred_class_1), pred_class_1)
    return prediction


eo_loader = DataLoader(BATCH_SIZE=BATCH_SIZE)
eo_loader.load_data(BATCH_SIZE=BATCH_SIZE)
loader = DataLoader(BATCH_SIZE=BATCH_SIZE, data_dir=PATH_TO_SAR + '/trainB/train')
loader.load_data(BATCH_SIZE=BATCH_SIZE)


def plot(out, name):
    plt.figure()
    plt.title(name)
    # img = rearrange(out, 'c h w -> h w c').cpu().detach().numpy()
    # plt.imshow((img - np.min(img)) / (np.max(img) - np.min(img)))
    plt.imshow(out)
    plt.colorbar()
    plt.clim(0, 250)
    plt.show()


start = time.time()
for epoch in range(EPOCHS):
    batch = np.expand_dims(loader.get_batch()[:,:,:,0], axis=-1)
    trg = np.expand_dims(eo_loader.get_batch()[:,:,:,0], axis=-1)

    output_img = np.array(conv_generator_train_step(batch, trg))
    output_img = conv_discriminator_train_step(batch, trg)

    if epoch % 10 == 0:
        print(
            f'Epoch : {epoch + 1}, '
            f'Generator Loss : {gen_loss_metric.result()}, '
            f'Discriminator Loss: {gen_loss_metric.result()}, '
            f'Runtime : {time.time() - start}, '
        )
        plot(output_img[0,:,:,0], 'output')
        plot(trg[0,:,:,0], 'target')

        # reset statistical measures
        start = time.time()
        gen_loss_metric.reset_states()
        dis_loss_metric.reset_states()

