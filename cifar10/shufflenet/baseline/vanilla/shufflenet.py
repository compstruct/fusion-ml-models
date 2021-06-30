#!/usr/bin/env python
# coding: utf-8

#####################################################################################################
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
TOTAL_GPUS = 3
ACTIVE_GPUS = 2
IMG_SIZE = IMG_H = IMG_W = 32
BATCH_SIZE = 96*ACTIVE_GPUS
EPOCHS = 126
CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.1
WDECAY = 1e-4
INIT_LEARNING_RATE = 1e-4
training = False
pre_trained = True
model_name = "vanilla.h5"


DATA_DIR = sys.argv[1]
training = sys.argv[2] == "True"
pre_trained = sys.argv[3] == "True"
TOTAL_GPUS = int(sys.argv[4])
ACTIVE_GPUS = TOTAL_GPUS
#####################################################################################################
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
    else:
        print("nothing avialable!")

setup_gpus(mem_alloc=1e4, num_gpu=ACTIVE_GPUS)

#Fetching, pre-processing & preparing data-pipeline
def preprocess(ds):
    x = tf.image.resize_with_pad(ds['image'], IMG_SIZE, IMG_SIZE)
    x = tf.cast(x, tf.float32)
    x = (x-120.70756512369792)/(64.1500758911213)
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
def ch_shuffle(x, pix, in_ch, groups):
    ch_per_group = in_ch // groups
    x_reshaped = tf.keras.layers.Reshape((pix, pix, groups, ch_per_group))(x)
    x_transposed = tf.keras.layers.Permute((1,2,4,3))(x_reshaped)
    x_shuffled = tf.keras.layers.Reshape((pix, pix, in_ch))(x_transposed)
    return x_shuffled

def group_conv(input, in_ch, out_ch, groups):
    in_group_ch = in_ch // groups
    out_group_ch = out_ch // groups
    tmp_list = []

    for group in range(groups):
        group_index = group * in_group_ch
        tmp_list.append(tf.keras.layers.Conv2D(filters=out_group_ch, kernel_size=(1,1),\
           padding='same', use_bias=False, kernel_initializer="he_normal",\
           kernel_regularizer=regularizers.l2(WDECAY))(input[:,:,:,group_index:group_index+in_group_ch]))

    return tf.keras.layers.Concatenate()(tmp_list)


def ShuffleUnit(input, in_ch, out_ch, groups=3, expansion=0.25, stride=1, pix=32):
    midd_ch = int(in_ch * expansion)

    pw1 = group_conv(input, in_ch=in_ch, out_ch=midd_ch, groups=groups)
    pw1_bn = tf.keras.layers.BatchNormalization()(pw1)
    pw1_rl = tf.keras.layers.ReLU()(pw1_bn)
    pw1_shuffled = ch_shuffle(pw1_rl, pix, midd_ch, groups)

    dw = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), padding='same', strides = stride, use_bias=False, kernel_initializer="he_normal")(pw1_shuffled)
    dw_bn = tf.keras.layers.BatchNormalization()(dw)

    pw2 = group_conv(dw_bn, in_ch=midd_ch, out_ch=(out_ch if stride == 1 and in_ch == out_ch else out_ch-in_ch), groups=groups)
    pw2_bn = tf.keras.layers.BatchNormalization()(pw2)

    if stride == 1 and in_ch == out_ch:
        result = tf.math.add(pw2_bn, input)
    else:
        residual = tf.keras.layers.AveragePooling2D(pool_size=3, strides=stride, padding='same')(input)
        result = tf.keras.layers.Concatenate()([pw2_bn, residual])
    return tf.keras.layers.ReLU()(result)



def build_model():
    inp = tf.keras.Input(shape=(32,32,3))
    Conv1 = tf.keras.layers.Conv2D(filters=24, kernel_size=(3,3), padding='same', use_bias=False, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(WDECAY))(inp)
    Conv1_bn = tf.keras.layers.BatchNormalization()(Conv1)
    Conv1_rl = tf.keras.layers.ReLU()(Conv1_bn)

    block0 = ShuffleUnit(input=Conv1_rl, in_ch=24, out_ch=240, expansion=1, pix=32)
    block1 = ShuffleUnit(input=block0, in_ch=240, out_ch=240, pix=32)
    block2 = ShuffleUnit(input=block1, in_ch=240, out_ch=240, pix=32)
    block3 = ShuffleUnit(input=block2, in_ch=240, out_ch=240, pix=32)

    block4 = ShuffleUnit(input=block3, in_ch=240, out_ch=480, stride=2, pix=32)
    block5 = ShuffleUnit(input=block4, in_ch=480, out_ch=480, pix=16)
    block6 = ShuffleUnit(input=block5, in_ch=480, out_ch=480, pix=16)
    block7 = ShuffleUnit(input=block6, in_ch=480, out_ch=480, pix=16)
    block8 = ShuffleUnit(input=block7, in_ch=480, out_ch=480, pix=16)
    block9 = ShuffleUnit(input=block8, in_ch=480, out_ch=480, pix=16)
    block10 = ShuffleUnit(input=block9, in_ch=480, out_ch=480, pix=16)
    block11 = ShuffleUnit(input=block10, in_ch=480, out_ch=480, pix=16)

    block12 = ShuffleUnit(input=block11, in_ch=480, out_ch=960, stride=2, pix=16)
    block13 = ShuffleUnit(input=block12, in_ch=960, out_ch=960, pix=8)
    block14 = ShuffleUnit(input=block13, in_ch=960, out_ch=960, pix=8)
    block15 = ShuffleUnit(input=block14, in_ch=960, out_ch=960, pix=8)


    global_ave_pool = tf.keras.layers.GlobalAveragePooling2D()(block15)
    dense = tf.keras.layers.Dense(10, activation='softmax', use_bias=True, kernel_initializer = 'glorot_uniform',bias_initializer = 'zeros', kernel_regularizer=regularizers.l2(WDECAY))(global_ave_pool)
    return tf.keras.models.Model(inputs=inp, outputs=dense)

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

    for i in range(120):
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
