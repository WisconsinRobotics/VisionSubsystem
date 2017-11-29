# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 17:06:01 2017

@author: tushar
"""
import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)
print(node1, node2)
print(sess.run([node1, node2]))
