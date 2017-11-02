# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 17:06:01 2017

@author: tushar
"""
import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

