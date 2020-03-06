#! /usr/bin/python3
# coding:utf-8

#pip install graphic-verification-code

from captcha.image import ImageCaptcha
import random,gvcode
import numpy as np
import tensorflow as tf


number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 128
CHAR_SET = number + alphabet + ALPHABET
CAPTCHA_SIZE = 4

def captcha_cnn():
    input_tensor = tf.keras.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))

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

class batchpic(object):
    def __init__(self, char_set, batch_size, captcha_size, batch_type='train_data'):
        self.char_set = ''.join(char_set)
        self.batch_size = batch_size
        self.captcha_size = captcha_size
        self.captchalist = self._random_captcha_list()
        self.batch_type = batch_type

    def _random_captcha_list(self):
        captcha = set()
        while len(captcha) < self.batch_size:
            random_str = ''.join([random.choice(self.char_set) for j in range(self.captcha_size)])
            captcha.add(random_str)
        return list(captcha)

    def _createpicbyImageCaptcha(self, chars):
        generator=ImageCaptcha(width=IMAGE_WIDTH,height=IMAGE_HEIGHT)
        img = generator.generate_image(chars)
        return img, chars

    def _createpicbygvcode(self):
        return gvcode.generate(size=(IMAGE_WIDTH,IMAGE_HEIGHT))

    def getpatches(self):
        batch_x = np.zeros((self.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        batch_y = [np.zeros((self.batch_size, len(self.char_set)), dtype=np.uint8) for i in range(self.captcha_size)]

        for i in range(self.batch_size):
            if self.batch_type == 'train_data':
                x, y = self._createpicbyImageCaptcha(self.captchalist[i])
            else:
                x, y = self._createpicbygvcode()
                
            x = np.array(x, 'd') 
            x = tf.convert_to_tensor(x)
            x /= 255.
            x = tf.reshape(x, (IMAGE_HEIGHT, IMAGE_WIDTH, 3))

            batch_x[i, :] = x
            for j, ch in enumerate(y):
                batch_y[j][i, :] = 0
                batch_y[j][i, self.char_set.index(ch)] = 1

        return batch_x, batch_y

    def vec2text(self, vec):
        text = []
        for item in vec:
            index = item[0]
            text.append(self.char_set[index])
        return ''.join(text)

class TrainAndPredict(object):
    def __init__(self, modelpath, batch_size, charset, captcha_size, epochs):
        self.model = captcha_cnn()
        self.modelpath = modelpath
        try:
            self.model.load_weights(self.modelpath + 'captcha_cnn_best.h5')
            self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-4, amsgrad=True),
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
        except Exception as identifier:
            print(identifier)
            self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-3, amsgrad=True),
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

        self.callbacks = [tf.keras.callbacks.EarlyStopping(patience=3),
                            tf.keras.callbacks.CSVLogger(self.modelpath + 'log/captcha_cnn.csv', append=True), 
                            tf.keras.callbacks.ModelCheckpoint(self.modelpath + 'captcha_cnn_best.h5', save_best_only=True)]
        
        self.batch_size = batch_size
        self.charset = charset
        self.captcha_size = captcha_size
        self.epochs = epochs

    def train(self, times):

        #train_type = 'train_data' if times % 2 == 0 else 'diff_data'
        train_type = 'train_data'

        train_data = batchpic(self.charset, self.batch_size, self.captcha_size, train_type)
        validation_data = batchpic(self.charset, 100, self.captcha_size, train_type)

        train_images, train_labels = train_data.getpatches()
        test_images, test_labels = validation_data.getpatches()

        self.model.fit(train_images, train_labels, epochs=self.epochs, 
                        validation_data=(test_images, test_labels), workers=4, use_multiprocessing=True,
                        callbacks=self.callbacks)

        if times % 100 == 0:
            print('times:', times)
            self.predict()

    def predict(self):

        success, succ, count = 0, 0, 100
        print('ing...')
        for _ in range(count):
            test_data = batchpic(self.charset, 1, self.captcha_size)

            data_x, data_y = test_data.getpatches()
            prediction_value = self.model.predict(data_x)

            data_y = test_data.vec2text(np.argmax(data_y, axis=2))
            prediction_value = test_data.vec2text(np.argmax(prediction_value, axis=2))
            success += 1 if data_y.upper() == prediction_value.upper() else 0


            ########################


            diff_test_data = batchpic(self.charset, 1, self.captcha_size, 'diff_data')

            diff_data_x, diff_data_y = diff_test_data.getpatches()
            diff_prediction_value = self.model.predict(diff_data_x)

            diff_data_y = diff_test_data.vec2text(np.argmax(diff_data_y, axis=2))
            diff_prediction_value = diff_test_data.vec2text(np.argmax(diff_prediction_value, axis=2))

            succ += 1 if diff_data_y.upper() == diff_prediction_value.upper() else 0

        print('captcha 数据(', count, '次)预测', '成功率 ：{:5.2%}'.format(success / count))
        print('gvcode 数据(', count, '次)预测', '成功率 ：{:5.2%}'.format(succ / count))

if __name__ == '__main__':
    MODEL_PATH = "/content/drive/APP/keras_cnn/"
    #MODEL_PATH = './keras_cnn/'

    BATCH_SIZE = 1024
    EPOCHS = 100

    cacnn = TrainAndPredict(MODEL_PATH, BATCH_SIZE, CHAR_SET, CAPTCHA_SIZE, EPOCHS)
    for i in range(1000):
        print('times:', i)
        cacnn.train(i)

    cacnn.predict()