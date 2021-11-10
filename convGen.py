import tensorflow as tf


class ConvGen(tf.keras.Model):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super(ConvGen, self).__init__()
        self.depth = 3
        self.linear_dim = int((image_size / (2 ** self.depth)) ** 2)

        self.Conv1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=2, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1, padding='same'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.ReLU()
        ])
        self.Conv2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=2, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1, padding='same'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.ReLU()
        ])
        self.Conv3 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=2, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=1, padding='same'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.ReLU()
        ])
        self.Conv4 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=1, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=1, padding='same'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.ReLU()
        ])
        self.Conv5 = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(128, kernel_size=(3, 3), strides=2, padding='same', output_padding=(1, 1)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1, padding='same'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.ReLU()
        ])
        self.Conv6 = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(64, kernel_size=(3, 3), strides=2, padding='same', output_padding=(1, 1)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1, padding='same'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.ReLU()
        ])
        self.Conv7 = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=2, padding='same', output_padding=(1, 1)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=1, padding='same'),
            tf.keras.layers.ReLU()
        ])
        self.Conv8 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, kernel_size=(3, 3), strides=1, padding='same'),
            tf.keras.layers.Conv2D(1, kernel_size=(3, 3), strides=1, padding='same')
        ])

    def call(self, x, **kwargs):
        c1 = self.Conv1(x)
        c2 = self.Conv2(c1)
        c3 = self.Conv3(c2)
        c4 = tf.concat((self.Conv4(c3), c3), axis=-1)
        c5 = tf.concat((self.Conv5(c4), c2), axis=-1)
        c6 = tf.concat((self.Conv6(c5), c1), axis=-1)
        c7 = tf.concat((self.Conv7(c6), x), axis=-1)
        return self.Conv8(c7)
