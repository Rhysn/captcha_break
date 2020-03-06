#! /usr/bin/python3
# coding:utf-8


import numpy as np
import tensorflow as tf
import batchpic as bp
import random


number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 128
CHAR_SET = number + alphabet + ALPHABET
CAPTCHA_SIZE = 4

def captcha_cnn(image_height=64, image_width=128):
    input_tensor = tf.keras.Input(shape=(image_height, image_width, 3))

    x = input_tensor
    for i, n in enumerate([2,2,2,2]):
        for _ in range(n):
            x = tf.keras.layers.Conv2D(32*2**min(i, 3), kernel_size=3, padding='same', activation='relu', kernel_initializer='he_uniform')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPool2D((2, 2), strides=2)(x)
        x = tf.keras.layers.Dropout(0.2)(x) if i == 0 or i == 3 else x

    x = tf.keras.layers.Flatten()(x)
    output_tensor = [tf.keras.layers.Dense(len(CHAR_SET), activation='softmax', name='c%d'%(i+1))(x) for i in range(CAPTCHA_SIZE)]

    model = tf.keras.Model(input_tensor, output_tensor)

    #model.summary()

    return model



class TrainAndPredict(object):
    def __init__(self, modelpath, batch_size, charset, captcha_size, epochs):
        self.model = captcha_cnn()
        self.modelpath = modelpath
        try:
            self.model.load_weights(self.modelpath + 'cnn_best.h5')
            self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-4, amsgrad=True),
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
        except Exception as identifier:
            print(identifier)
            self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-3, amsgrad=True),
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

        self.callbacks = [tf.keras.callbacks.EarlyStopping(patience=3),
                            tf.keras.callbacks.CSVLogger(self.modelpath + 'log/cnn.csv', append=True), 
                            tf.keras.callbacks.ModelCheckpoint(self.modelpath + 'cnn_best.h5', save_best_only=True)]
        
        self.batch_size = batch_size
        self.charset = charset
        self.captcha_size = captcha_size
        self.epochs = epochs

    def train(self):

        train_type = random.choice(['train_data','diff_data'])

        train_data = bp.batchpic(self.charset, self.batch_size, self.captcha_size, train_type)
        validation_data = bp.batchpic(self.charset, 100, self.captcha_size, train_type)

        train_images, train_labels = train_data.getpatches()
        test_images, test_labels = validation_data.getpatches()

        self.model.fit(train_images, train_labels, epochs=self.epochs, 
                        validation_data=(test_images, test_labels), workers=4, use_multiprocessing=True,
                        callbacks=self.callbacks)

    def predict(self):

        success, succ, count = 0, 0, 100
        print('Runing...')
        for _ in range(count):
            test_data = bp.batchpic(self.charset, 1, self.captcha_size)

            data_x, data_y = test_data.getpatches()
            prediction_value = self.model.predict(data_x)

            data_y = test_data.vec2text(np.argmax(data_y, axis=2))
            prediction_value = test_data.vec2text(np.argmax(prediction_value, axis=2))
            success += 1 if data_y.upper() == prediction_value.upper() else 0


            ########################


            diff_test_data = bp.batchpic(self.charset, 1, self.captcha_size, 'diff_data')

            diff_data_x, diff_data_y = diff_test_data.getpatches()
            diff_prediction_value = self.model.predict(diff_data_x)

            diff_data_y = diff_test_data.vec2text(np.argmax(diff_data_y, axis=2))
            diff_prediction_value = diff_test_data.vec2text(np.argmax(diff_prediction_value, axis=2))

            succ += 1 if diff_data_y.upper() == diff_prediction_value.upper() else 0

        print('captcha 数据(', count, '次)预测', '成功率 ：{:5.2%}'.format(success / count))
        print('gvcode 数据(', count, '次)预测', '成功率 ：{:5.2%}'.format(succ / count))

if __name__ == '__main__':
    #MODEL_PATH = "/content/drive/APP/keras_cnn/"
    MODEL_PATH = './keras_cnn/'

    BATCH_SIZE = 1000
    EPOCHS = 100

    cacnn = TrainAndPredict(MODEL_PATH, BATCH_SIZE, CHAR_SET, CAPTCHA_SIZE, EPOCHS)
    for _ in range(1000):
        cacnn.train()

    cacnn.predict()