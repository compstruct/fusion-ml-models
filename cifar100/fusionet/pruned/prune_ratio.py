import sys
sys.path = ['', '/usr/lib/python36.zip', '/usr/lib/python3.6', '/usr/lib/python3.6/lib-dynload', '/home/mohamadol/.local/lib/python3.6/site-packages',
            '/usr/local/lib/python3.6/dist-packages']
import numpy as np
import tensorflow as tf


#model = tf.keras.models.load_model("./post_prune.h5")
model = tf.keras.models.load_model("./pruned_named.h5")
layers = model.layers
model.summary()


conv_number = 0
depthwise_number = 0
dense_number = 0
for layer in layers:
    if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
        tmp = layer.get_weights()[0]
        mask =  np.absolute(tmp) !=0
        prune_ratio = np.sum(mask) / np.size(tmp)
        print("dwise_" + str(depthwise_number) + " : " + str(prune_ratio))
        depthwise_number += 1
    elif isinstance(layer, tf.keras.layers.Conv2D):
        tmp = layer.get_weights()[0]
        mask =  np.absolute(tmp) !=0
        prune_ratio = np.sum(mask) / np.size(tmp)
        print("conv_" + str(conv_number) + " : " + str(prune_ratio))
        conv_number += 1
    elif isinstance(layer, tf.keras.layers.Dense):
        tmp = layer.get_weights()[0]
        mask =  np.absolute(tmp) !=0
        prune_ratio = np.sum(mask) / np.size(tmp)
        print("dense_" + str(dense_number) + " weight : " + str(prune_ratio))
        tmp = layer.get_weights()[1]
        mask =  np.absolute(tmp) !=0
        prune_ratio = np.sum(mask) / np.size(tmp)
        print("dense_" + str(dense_number) + "bias : " + str(prune_ratio))
        dense_number += 1


