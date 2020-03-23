import tensorflow as tf
from tensorflow import keras

def VGG16(im_height=224, im_width=224, class_num=1000):
    model = keras.models.Sequential([
        keras.layers.Conv2D(64, kernel_size=3, padding="SAME", activation="relu"),
        keras.layers.Conv2D(64, kernel_size=3, padding="SAME", activation="relu"),
        keras.layers.MaxPool2D(pool_size=2, strides=2),

        keras.layers.Conv2D(128, kernel_size=3, padding="SAME", activation="relu"),
        keras.layers.Conv2D(128, kernel_size=3, padding="SAME", activation="relu"),
        keras.layers.MaxPool2D(pool_size=2, strides=2),

        keras.layers.Conv2D(256, kernel_size=3, padding="SAME", activation="relu"),
        keras.layers.Conv2D(256, kernel_size=3, padding="SAME", activation="relu"),
        keras.layers.Conv2D(256, kernel_size=3, padding="SAME", activation="relu"),
        keras.layers.MaxPool2D(pool_size=2, strides=2),

        keras.layers.Conv2D(512, kernel_size=3, padding="SAME", activation="relu"),
        keras.layers.Conv2D(512, kernel_size=3, padding="SAME", activation="relu"),
        keras.layers.Conv2D(512, kernel_size=3, padding="SAME", activation="relu"),
        keras.layers.MaxPool2D(pool_size=2, strides=2),

        keras.layers.Conv2D(512, kernel_size=3, padding="SAME", activation="relu"),
        keras.layers.Conv2D(512, kernel_size=3, padding="SAME", activation="relu"),
        keras.layers.Conv2D(512, kernel_size=3, padding="SAME", activation="relu"),
        keras.layers.MaxPool2D(pool_size=2, strides=2),

        keras.layers.Flatten(),
        keras.layers.Dropout(rate=0.5),
        keras.layers.Dense(2048, activation="relu"),
        keras.layers.Dropout(rate=0.5),
        keras.layers.Dense(2048, activation="relu"),
        keras.layers.Dense(class_num),
        keras.layers.Softmax(),
    ])
    return model
