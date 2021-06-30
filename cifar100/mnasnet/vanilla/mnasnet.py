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
DATASET = 'cifar100'
DATA_DIR = "/home/mohamadol/tensorflow_datasets"
log_dir_parent = "tb/"
TOTAL_GPUS = 4
ACTIVE_GPUS = 4
IMG_SIZE = IMG_H = IMG_W = 32
BATCH_SIZE = 16 * ACTIVE_GPUS
EPOCHS = 140
CLASSES = 100
VERBOSE = 1
WDECAY = 4e-5
INIT_LEARNING_RATE = 1e-4

training = False
pre_trained = True

model_name = "vanilla.h5"

DATA_DIR = sys.argv[1]
training = sys.argv[2] == "True"
pre_trained = sys.argv[3] == "True"
TOTAL_GPUS = int(sys.argv[4])
ACTIVE_GPUS = TOTAL_GPUS
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

setup_gpus(mem_alloc=9.5e3, num_gpu=ACTIVE_GPUS)

#Fetching, pre-processing & preparing data-pipeline
def preprocess(ds):
    x = tf.image.resize_with_pad(ds['image'], IMG_SIZE, IMG_SIZE)
    x = tf.cast(x, tf.float32)
    x = (x-121.90086)/(68.23722)
    y = tf.one_hot(ds['label'], CLASSES)
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
def MNAS_block(inp, in_ch, midd_ch, out_ch, kernel_size, stride=1, SE=False, SE_ratio=0.25):

    decompress = tf.keras.layers.Conv2D(filters=midd_ch, kernel_size=1, padding='same', use_bias=False, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(WDECAY))(inp)
    decompress_BN = tf.keras.layers.BatchNormalization()(decompress)
    decompress_ReLU = tf.keras.layers.ReLU()(decompress_BN)

    convdw = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding='same', use_bias=False, kernel_initializer="he_normal")(decompress_ReLU)
    convdw_BN = tf.keras.layers.BatchNormalization()(convdw)
    convdw_ReLU = tf.keras.layers.ReLU()(convdw_BN)

    if SE:
        squeeze = tf.keras.layers.GlobalAveragePooling2D()(convdw_ReLU)
        squeeze1 = tf.expand_dims(squeeze, 1)
        squeeze2 = tf.expand_dims(squeeze1, 1)
        excite1 = tf.keras.layers.Conv2D(filters=midd_ch*SE_ratio, kernel_size=1, padding='same', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(WDECAY))(squeeze2)
        excite1_rl = tf.keras.layers.ReLU()(excite1)
        excite2 = tf.keras.layers.Conv2D(filters=midd_ch, kernel_size=1, padding='same', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(WDECAY))(excite1_rl)
        excitation = tf.math.sigmoid(excite2)
        dw_excited = tf.math.multiply(excitation, convdw_ReLU)
    else:
        dw_excited = convdw_ReLU

    compress = tf.keras.layers.Conv2D(filters=out_ch, kernel_size=1, padding='same', use_bias=False, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(WDECAY))(dw_excited)
    compress_bn = tf.keras.layers.BatchNormalization()(compress)

    if stride == 1 and in_ch==out_ch:
        res = tf.math.add(compress_bn, inp)
    else:
        res = compress_bn
    return res


#*****************************************************************************************************
def build_model():

    inp = tf.keras.Input(shape=(32,32,3))
    Conv1 = tf.keras.layers.Conv2D(32, (3,3),padding='same', use_bias=False, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(WDECAY))(inp)
    bn_Conv1 = tf.keras.layers.BatchNormalization()(Conv1)
    relu_Conv1 = tf.keras.layers.ReLU()(bn_Conv1)

    expanded_conv_depthwise = tf.keras.layers.DepthwiseConv2D((3,3), padding='same', use_bias=False, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(WDECAY))(relu_Conv1)
    expanded_conv_depthwise_BN = tf.keras.layers.BatchNormalization()(expanded_conv_depthwise)
    expanded_conv_depthwise_relu = tf.keras.layers.ReLU()(expanded_conv_depthwise_BN)
    expanded_conv_project = tf.keras.layers.Conv2D(16,(1,1),padding='same', use_bias=False, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(WDECAY))(expanded_conv_depthwise_relu)
    b0 = tf.keras.layers.BatchNormalization()(expanded_conv_project)

    b1 = MNAS_block(b0, 16, 16*6, 24, kernel_size=3)
    b2 = MNAS_block(b1, 24, 24*6, 24, kernel_size=3)

    b3 = MNAS_block(b2, 24, 24*3, 40, kernel_size=5, stride=2, SE=True)
    b4 = MNAS_block(b3, 40, 40*3, 40, kernel_size=5, SE=True)
    b5 = MNAS_block(b4, 40, 40*3, 40, kernel_size=5, SE=True)

    b6 = MNAS_block(b5, 40, 40*6, 80, kernel_size=3)
    b7 = MNAS_block(b6, 80, 80*6, 80, kernel_size=3)
    b8 = MNAS_block(b7, 80, 80*6, 80, kernel_size=3)
    b9 = MNAS_block(b8, 80, 80*6, 80, kernel_size=3)

    b10 = MNAS_block(b9, 80, 80*6, 112, kernel_size=3, SE=True)
    b11 = MNAS_block(b10, 112, 112*6, 112, kernel_size=3, SE=True)

    b12 = MNAS_block(b11, 112, 112*6, 160, kernel_size=5, stride=2, SE=True)
    b13 = MNAS_block(b12, 160, 160*6, 160, kernel_size=5, SE=True)
    b14 = MNAS_block(b13, 160, 160*6, 160, kernel_size=5, SE=True)

    b15 = MNAS_block(b14, 160, 160*6, 320, kernel_size=3)

    expanded_lastConv = tf.keras.layers.Conv2D(filters=1280, kernel_size=(1,1), padding='same', use_bias=False, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(WDECAY))(b15)
    #pw_final = group_conv(block16, in_ch=160, out_ch=1280, groups=4)
    final_bn = tf.keras.layers.BatchNormalization()(expanded_lastConv)
    final_rl = tf.keras.layers.ReLU()(final_bn)

    global_ave_pool = tf.keras.layers.GlobalAveragePooling2D()(final_rl)
    dense = tf.keras.layers.Dense(100, activation='softmax', use_bias=True, kernel_initializer = 'glorot_uniform',bias_initializer = 'zeros', kernel_regularizer=regularizers.l2(WDECAY))(global_ave_pool)

    model = tf.keras.models.Model(inputs=inp, outputs=dense)
    return model



#####################################################################################################
#Learing rate & checkpoints
def generate_learning_rates():
    learning_rates=[]
    lr = 0.096
    decay_steps = 140
    alpha = 0.0004

    for i in range(EPOCHS):
        steps = min(decay_steps, i)
        cosine_decay = 0.25 * (1 + np.cos( np.pi * steps/decay_steps))
        lr = (1 - alpha) * cosine_decay + alpha
        learning_rates.append(lr)

    return learning_rates

def generate_checkpoint(model_name):
    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_name,
                monitor='val_accuracy', verbose=1, save_best_only=True,\
                save_weights_only=False, mode='auto', save_freq='epoch')
    return checkpoint

def get_callbacks():
    learning_rates = generate_learning_rates()
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
