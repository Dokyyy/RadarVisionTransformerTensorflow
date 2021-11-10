import tensorflow as tf
import numpy as np
from configuration import *
from convGen import *
from dataloader import *

model = ConvGen(
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

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def conv_gen_train_step(inputs, output):
    with tf.GradientTape() as tape:
        prediction = model(inputs)
        pix_loss = pixel_mse(output, prediction)

    gradients = tape.gradient(pix_loss, model.trainable_variables)
    pixel_opt.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(pix_loss)
    # train_accuracy(output, prediction)
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


while True:

    batch_num = 0
    batch = loader.get_batch()
    trg = eo_loader.get_batch()

    output_img = np.array(conv_gen_train_step(batch, trg))

    if batch_num % 10 == 0:
        plot(output_img[0,:,:,0], 'output')
        plot(trg[0,:,:,0], 'target')
    batch_num += 1


