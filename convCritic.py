import tensorflow as tf


class ConvCritic(tf.keras.Model):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super(ConvCritic, self).__init__()
        self.depth = 3
        self.linear_dim = int((image_size / (2 ** self.depth)) ** 2)

        self.Conv1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=2, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1, padding='same'),
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
            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=2, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1, padding='same'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.ReLU()
        ])
        self.Conv4 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=2, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1, padding='same'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.ReLU()
        ])
        self.Conv5 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=2, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1, padding='same'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.ReLU()
        ])
        self.Conv6 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=2, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1, padding='same'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.ReLU()
        ])
        self.Conv7 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=2, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1, padding='same'),
            tf.keras.layers.ReLU()
        ])
        self.Conv8 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=2, padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1)
        ])

        self.max_pool = tf.keras.layers.MaxPool2D()

    def call(self, x, **kwargs):
        c1 = self.Conv1(x)
        c2 = self.Conv2(c1) + self.max_pool(c1)
        c3 = self.Conv3(c2) + self.max_pool(c2)
        c4 = self.Conv4(c3) + self.max_pool(c3)
        c5 = self.Conv5(c4) + self.max_pool(c4)
        c6 = self.Conv6(c5) + self.max_pool(c5)
        c7 = self.Conv7(c6) + self.max_pool(c6)
        return self.Conv8(c7)
