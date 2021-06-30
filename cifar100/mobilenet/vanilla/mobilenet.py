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
DATASET = 'cifar100'
DATA_DIR = "/home/mohamadol/tensorflow_datasets"
log_dir_parent = "tb/"

TOTAL_GPUS = 4
ACTIVE_GPUS = 4 #3
GPU_MEM = 9.8e3
CPUs = 8

IMG_SIZE = IMG_H = IMG_W = 32
BATCH_SIZE = 16*ACTIVE_GPUS
EPOCHS = 140
CLASSES = 100
VERBOSE = 1

VALIDATION_SPLIT = 0.1
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

setup_gpus(mem_alloc=GPU_MEM, num_gpu=ACTIVE_GPUS)

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

    train = train.repeat().shuffle(shuffle_buff_size).map(preprocess, num_parallel_calls=CPUs).batch(batch_size).map(augmentation, num_parallel_calls=CPUs)
    train = train.prefetch(tf.data.experimental.AUTOTUNE)

    val = val.map(preprocess).cache().repeat().batch(batch_size)
    val = val.prefetch(tf.data.experimental.AUTOTUNE)
    return train, info_train, val, info_val, TRAIN_SIZE, VAL_SIZE

#####################################################################################################
def InvertedBottleneck(input, in_ch, out_ch, expansion=6, stride=1):
    midd_ch = in_ch * expansion
    decompress = tf.keras.layers.Conv2D(filters=midd_ch, kernel_size=(1,1), padding='same', use_bias=False, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(WDECAY))(input)
    decompress_bn = tf.keras.layers.BatchNormalization()(decompress)
    decompress_rl = tf.keras.layers.ReLU()(decompress_bn)

    convdw = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), padding='same', strides = stride, use_bias=False, kernel_initializer="he_normal")(decompress_rl)
    convdw_bn = tf.keras.layers.BatchNormalization()(convdw)
    convdw_rl = tf.keras.layers.ReLU()(convdw_bn)

    compress = tf.keras.layers.Conv2D(filters=out_ch, kernel_size=(1,1), padding='same', use_bias=False, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(WDECAY))(convdw_rl)
    compress_bn = tf.keras.layers.BatchNormalization()(compress)

    if in_ch == out_ch and stride == 1:
        result = tf.math.add(compress_bn, input)
    else:
        result = compress_bn
    return result



def build_model():
    inp = tf.keras.Input(shape=(32,32,3))
    Conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', use_bias=False, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(WDECAY))(inp)
    Conv1_bn = tf.keras.layers.BatchNormalization()(Conv1)
    Conv1_rl = tf.keras.layers.ReLU()(Conv1_bn)

    block0 = InvertedBottleneck(input=Conv1_rl, in_ch=32, out_ch=16, expansion=1)
    block1 = InvertedBottleneck(input=block0, in_ch=16, out_ch=24)
    block2 = InvertedBottleneck(input=block1, in_ch=24, out_ch=24)

    block3 = InvertedBottleneck(input=block2, in_ch=24, out_ch=32)
    block4 = InvertedBottleneck(input=block3, in_ch=32, out_ch=32)
    block5 = InvertedBottleneck(input=block4, in_ch=32, out_ch=32)

    block6 = InvertedBottleneck(input=block5, in_ch=32, out_ch=64, stride=2)
    block7 = InvertedBottleneck(input=block6, in_ch=64, out_ch=64)
    block8 = InvertedBottleneck(input=block7, in_ch=64, out_ch=64)
    block9 = InvertedBottleneck(input=block8, in_ch=64, out_ch=64)

    block10 = InvertedBottleneck(input=block9, in_ch=64, out_ch=96)
    block11 = InvertedBottleneck(input=block10, in_ch=96, out_ch=96)
    block12 = InvertedBottleneck(input=block11, in_ch=96, out_ch=96)

    block13 = InvertedBottleneck(input=block12, in_ch=96, out_ch=160, stride=2)
    block14 = InvertedBottleneck(input=block13, in_ch=160, out_ch=160)
    block15 = InvertedBottleneck(input=block14, in_ch=160, out_ch=160)

    block16 = InvertedBottleneck(input=block15, in_ch=160, out_ch=320)

    expanded_lastConv = tf.keras.layers.Conv2D(filters=1280, kernel_size=(1,1), padding='same', use_bias=False, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(WDECAY))(block16)
    expanded_lastConv_bn = tf.keras.layers.BatchNormalization()(expanded_lastConv)
    expanded_lastConv_rl = tf.keras.layers.ReLU()(expanded_lastConv_bn)
    global_ave_pool = tf.keras.layers.GlobalAveragePooling2D()(expanded_lastConv_rl)

    dense = tf.keras.layers.Dense(100, activation='softmax', use_bias=True, kernel_initializer = 'glorot_uniform',bias_initializer = 'zeros', kernel_regularizer=regularizers.l2(WDECAY))(global_ave_pool)
    return tf.keras.models.Model(inputs=inp, outputs=dense)

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
#####################################################################################################

main()
