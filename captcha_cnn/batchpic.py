#! /usr/bin/python3
# coding:utf-8

#pip install graphic-verification-code

import numpy as np
from captcha.image import ImageCaptcha
import random,gvcode
import tensorflow as tf

class batchpic(object):
    def __init__(self, char_set, batch_size, captcha_size, batch_type='train_data', image_width=128, image_height=64):
        self.char_set = ''.join(char_set)
        self.batch_size = batch_size
        self.captcha_size = captcha_size
        self.captchalist = self._random_captcha_list()
        self.batch_type = batch_type
        self.image_height = image_height
        self.image_width = image_width

    def _random_captcha_list(self):
        captcha = set()
        while len(captcha) < self.batch_size:
            random_str = ''.join([random.choice(self.char_set) for j in range(self.captcha_size)])
            captcha.add(random_str)
        return list(captcha)

    def _createpicbyImageCaptcha(self, chars):
        generator=ImageCaptcha(width=self.image_width,height=self.image_height)
        img = generator.generate_image(chars)
        return img, chars

    def _createpicbygvcode(self):
        return gvcode.generate(size=(self.image_width,self.image_height))

    def getpatches(self):
        batch_x = np.zeros((self.batch_size, self.image_height, self.image_width, 3))
        batch_y = [np.zeros((self.batch_size, len(self.char_set)), dtype=np.uint8) for i in range(self.captcha_size)]

        for i in range(self.batch_size):
            if self.batch_type == 'train_data':
                x, y = self._createpicbyImageCaptcha(self.captchalist[i])
            else:
                x, y = self._createpicbygvcode()

            x = np.array(x, 'd') 
            x = tf.convert_to_tensor(x)
            x /= 255.
            x = tf.reshape(x, (self.image_height, self.image_width, 3))

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