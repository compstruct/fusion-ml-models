#!/usr/bin/env python
# coding: utf-8

#####################################################################################################
#Import all packages & libraries
#https://github.com/nsarang/MnasNet/blob/master/MnasNet.py
import sys
sys.path = ['', sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9]]

import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
import datetime
tfds.disable_progress_bar()
#####################################################################################################



#####################################################################################################
#*****************Global Variables & setup*****************
DATASET = 'cifar10'
DATA_DIR = "/home/mohamadol/tensorflow_datasets"
log_dir_parent = "tb/"
TOTAL_GPUS = 2
ACTIVE_GPUS = 2
IMG_SIZE = IMG_H = IMG_W = 32
BATCH_SIZE = 96*ACTIVE_GPUS
EPOCHS = 150
CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.1
WDECAY = 1e-4
INIT_LEARNING_RATE = 1e-4
training = False
pre_trained = True
model_name = "vanilla.h5"
#**********************************************************

def setup_gpus(mem_alloc, num_gpu):
    GPU_begin = TOTAL_GPUS - ACTIVE_GPUS
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[GPU_begin:TOTAL_GPUS],'GPU')
            for gpu in range(GPU_begin, TOTAL_GPUS):
                tf.config.experimental.set_virtual_device_configuration(gpus[gpu], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=mem_alloc)])
        except RuntimeError as e:
            print(e)

setup_gpus(mem_alloc=9.4e3, num_gpu=ACTIVE_GPUS)


#Fetching, pre-processing & preparing data-pipeline
def preprocess(ds):
    x = tf.image.resize_with_pad(ds['image'], IMG_SIZE, IMG_SIZE)
    x = tf.cast(x, tf.float32)
    x = (x-120.70756512369792)/(64.1500758911213)
    y = tf.one_hot(ds['label'], 10)
    return x, y

def augmentation(image,label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.resize_with_crop_or_pad(image, IMG_W+4, IMG_W+4) # zero pad each side with 4 pixels
    image = tf.image.random_crop(image, size=[BATCH_SIZE, IMG_W, IMG_H, 3]) # Random crop back to 32x32
    return image, label

#Data-pipeline: Load dataset into 90% train and 10% validation
def get_dataset(dataset_name, shuffle_buff_size=1024, batch_size=BATCH_SIZE, augmented=True):
    train, info_train = tfds.load(dataset_name, split='train', with_info=True)
    val, info_val = tfds.load(dataset_name, split='test', with_info=True)

    TRAIN_SIZE = info_train.splits['train'].num_examples
    VAL_SIZE = info_train.splits['test'].num_examples

    train = train.repeat().shuffle(shuffle_buff_size).map(preprocess, num_parallel_calls=4).batch(batch_size).map(augmentation, num_parallel_calls=4)
    train = train.prefetch(tf.data.experimental.AUTOTUNE)

    val = val.map(preprocess).cache().repeat().batch(batch_size)
    val = val.prefetch(tf.data.experimental.AUTOTUNE)
    return train, info_train, val, info_val, TRAIN_SIZE, VAL_SIZE

#####################################################################################################
def MNAS_block(inp, in_ch, midd_ch, out_ch, kernel_size, stride=1, SE=False, SE_ratio=16, decay=WDECAY):

    decompress = tf.keras.layers.Conv2D(filters=midd_ch, kernel_size=1, padding='same', use_bias=False, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(decay))(inp)
    decompress_BN = tf.keras.layers.BatchNormalization()(decompress)
    decompress_ReLU = tf.keras.layers.ReLU()(decompress_BN)

    convdw = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding='same', use_bias=False, kernel_initializer="he_normal")(decompress_ReLU)
    convdw_BN = tf.keras.layers.BatchNormalization()(convdw)
    convdw_ReLU = tf.keras.layers.ReLU()(convdw_BN)


    compress = tf.keras.layers.Conv2D(filters=out_ch, kernel_size=1, padding='same', use_bias=False, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(decay))(convdw_ReLU)
    compress_bn = tf.keras.layers.BatchNormalization()(compress)

    if SE:
        squeeze = tf.keras.layers.GlobalAveragePooling2D()(compress_bn)
        squeeze1 = tf.expand_dims(squeeze, 1)
        squeeze2 = tf.expand_dims(squeeze1, 1)
        excite1 = tf.keras.layers.Conv2D(filters=out_ch*SE_ratio, kernel_size=1, padding='same', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(4e-5))(squeeze2)
        excite1_rl = tf.keras.layers.ReLU()(excite1)
        excite2 = tf.keras.layers.Conv2D(filters=out_ch, kernel_size=1, padding='same', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(4e-5))(excite1_rl)
        excitation = tf.math.sigmoid(excite2)
        com_excited = tf.math.multiply(excitation, compress_bn)
    else:
        com_excited = compress_bn

    if stride == 1 and in_ch==out_ch:
        res = tf.math.add(com_excited, inp)
    else:
        res = com_excited
    return res


#*****************************************************************************************************
def build_model():

    inp = tf.keras.Input(shape=(32,32,3))
    Conv1 = tf.keras.layers.Conv2D(32, (3,3),padding='same', use_bias=False, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(inp)
    bn_Conv1 = tf.keras.layers.BatchNormalization()(Conv1)
    relu_Conv1 = tf.keras.layers.ReLU()(bn_Conv1)

    expanded_conv_depthwise = tf.keras.layers.DepthwiseConv2D((3,3), padding='same', use_bias=False, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(relu_Conv1)
    expanded_conv_depthwise_BN = tf.keras.layers.BatchNormalization()(expanded_conv_depthwise)
    expanded_conv_depthwise_relu = tf.keras.layers.ReLU()(expanded_conv_depthwise_BN)
    expanded_conv_project = tf.keras.layers.Conv2D(16,(1,1),padding='same', use_bias=False, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(expanded_conv_depthwise_relu)
    b0 = tf.keras.layers.BatchNormalization()(expanded_conv_project)

    b1 = MNAS_block(b0, 16, 960, 24, kernel_size=3, decay=8e-4)
    b2 = MNAS_block(b1, 24, 960, 24, kernel_size=3, decay=8e-4)

    b3 = MNAS_block(b2, 24, 960, 24, kernel_size=5, stride=2, SE=True, decay=8e-4)
    b4 = MNAS_block(b3, 24, 960, 32, kernel_size=5, SE=True, decay=4e-4)
    b5 = MNAS_block(b4, 32, 960, 32, kernel_size=5, SE=True, decay=4e-4)

    b6 = MNAS_block(b5, 32, 960, 32, kernel_size=3, decay=4e-4)
    b7 = MNAS_block(b6, 32, 960, 32, kernel_size=3, decay=4e-4)
    b8 = MNAS_block(b7, 32, 960, 40, kernel_size=3, decay=4e-4)

    b9 = MNAS_block(b8, 32, 960, 40, kernel_size=3, stride=2, decay=4e-4)
    b10 = MNAS_block(b9, 40, 960, 40, kernel_size=3, SE=True, SE_ratio=64)
    b11 = MNAS_block(b10, 40, 960, 40, kernel_size=3, SE=True, SE_ratio=128)

    b12 = MNAS_block(b11, 40, 960, 40, kernel_size=5, SE=True, SE_ratio=64)
    b13 = MNAS_block(b12, 40, 960, 40, kernel_size=5, SE=True, SE_ratio=128)
    b14 = MNAS_block(b13, 40, 960, 40, kernel_size=5, SE=True, SE_ratio=128)

    b15 = MNAS_block(b14, 40, 960, 80, kernel_size=3)

    global_ave_pool = tf.keras.layers.GlobalAveragePooling2D()(b15)
    dense = tf.keras.layers.Dense(10, activation='softmax', use_bias=True, kernel_initializer = 'glorot_uniform',bias_initializer = 'zeros', kernel_regularizer=regularizers.l2(4e-5))(global_ave_pool)

    model = tf.keras.models.Model(inputs=inp, outputs=dense)
    return model



#####################################################################################################
#Learing rate & checkpoints

def generate_learning_rates1():
    learning_rates=[]
    lr = INIT_LEARNING_RATE
    decay_steps = 120
    alpha = 0.0008
    for i in range(6):
        if i % 2 == 0:
            lr *= 10
        learning_rates.append(lr)

    for i in range(144):
        steps = min(decay_steps, i)
        cosine_decay = 0.05 * (1 + np.cos( np.pi * steps/decay_steps))
        lr = (1 - alpha) * cosine_decay + alpha
        learning_rates.append(lr)

    return learning_rates


def generate_checkpoint(model_name):
    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_name,
                monitor='val_accuracy', verbose=1, save_best_only=True,\
                save_weights_only=False, mode='auto', save_freq='epoch')
    return checkpoint

def get_callbacks():
    learning_rates = generate_learning_rates1()
    MODEL_NAME = model_name
    log_dir = log_dir_parent + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint = generate_checkpoint(MODEL_NAME)
    callbacks = []
    callbacks.append(checkpoint)
    callbacks.append(tf.keras.callbacks.LearningRateScheduler(lambda epoch: float(learning_rates[epoch])))
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))
    return callbacks
#####################################################################################################



#####################################################################################################
#Main Function
LOSS = ['categorical_crossentropy']
ACCURACY = ['accuracy']

def main():

    global DATA_DIR, training, pre_trained, TOTAL_GPUS, ACTIVE_GPUS
    DATA_DIR = sys.argv[1]
    training = sys.argv[2] == "True"
    pre_trained = sys.argv[3] == "True"
    TOTAL_GPUS = int(sys.argv[4])
    ACTIVE_GPUS = TOTAL_GPUS

    with tf.device('/cpu:0'):
        train, info_train, val, info_val, train_size, val_size = get_dataset(DATASET, shuffle_buff_size=25*1024)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        if not pre_trained:
            model = build_model()
            optimizer = tf.keras.optimizers.SGD(learning_rate=INIT_LEARNING_RATE, momentum=0.9)
            model.compile(loss=LOSS, optimizer=optimizer, metrics=ACCURACY)
        else:
            model = tf.keras.models.load_model(model_name)
        if training:
            model.fit(train,
                epochs=EPOCHS,
                steps_per_epoch=int(train_size / BATCH_SIZE),
                validation_data=val,
                validation_steps=int(val_size / BATCH_SIZE),
                validation_freq=1,
                callbacks=get_callbacks())
        else:
            model.evaluate(val, steps=int(val_size/BATCH_SIZE))
            #model.summary()
#####################################################################################################

main()
