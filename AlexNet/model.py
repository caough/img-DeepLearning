from tensorflow import keras


class AlexNet(keras.Model):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = keras.Sequential([
            keras.layers.ZeroPadding2D(((1, 2), (1, 2))),
            keras.layers.Conv2D(48, kernel_size=11, strides=4, activation='relu'),
            keras.layers.MaxPool2D(pool_size=3, strides=2),
            keras.layers.Conv2D(128, kernel_size=5, padding='same', activation='relu'),
            keras.layers.MaxPool2D(pool_size=3, strides=2),
            keras.layers.Conv2D(192, kernel_size=3, padding='same', activation='relu'),
            keras.layers.Conv2D(192, kernel_size=3, padding='same', activation='relu'),
            keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'),
            keras.layers.MaxPool2D(pool_size=3, strides=2),

        ])
        self.flatten = keras.layers.Flatten()
        self.classifier = keras.Sequential([
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1024, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(num_classes),
            keras.layers.Softmax(),
        ])

    def call(self, inputs, **kwargs):
        x = self.features(inputs)
        x = self.flatten(x)
        x = self.classifier(x)
        return x