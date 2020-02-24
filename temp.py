import tensorflow as tf

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))

