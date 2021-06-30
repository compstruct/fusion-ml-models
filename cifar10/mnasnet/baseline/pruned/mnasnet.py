#!/usr/bin/env python
# coding: utf-8

#####################################################################################################
#Import all packages & libraries
#https://github.com/nsarang/MnasNet/blob/master/MnasNet.py
import sys
sys.path = ['', sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9]]
import time
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
vanilla_model = "./../vanilla/vanilla.h5"
pruned_model = "post_prune.h5"


#4 GPUs were used for training
TOTAL_GPUS = 2  #4
ACTIVE_GPUS = 2 #4
IMG_SIZE = IMG_H = IMG_W = 32
BATCH_SIZE = 64*ACTIVE_GPUS
EPOCHS = 80
CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.01
WDECAY = 1e-4
INIT_LEARNING_RATE = 1e-1
RECOVERY_EPOCHS = 50

PW_INIT_SPARSITY = 0.0
PW_FINAL_SPARSITY = 0.8

DW_INIT_SPARSITY = 0.0
DW_FINAL_SPARSITY = 0.7

training = False
pre_trained = True
train_ternary = True

vanilla_layers = []
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

setup_gpus(mem_alloc=10e3, num_gpu=ACTIVE_GPUS)

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

    train = train.repeat().shuffle(shuffle_buff_size).map(preprocess, num_parallel_calls=6).batch(batch_size).map(augmentation, num_parallel_calls=6)
    train = train.prefetch(tf.data.experimental.AUTOTUNE)

    val = val.map(preprocess).cache().repeat().batch(batch_size)
    val = val.prefetch(tf.data.experimental.AUTOTUNE)
    return train, info_train, val, info_val, TRAIN_SIZE, VAL_SIZE




#####################################################################################################







####################################################################################################
#Pruning********************************************************************************************
global prune_layers
prune_layers =[]
for i in range(2,48):
    if i not in [7, 8, 11, 12, 15, 16, 27, 28, 31, 32, 35, 36, 39, 40, 43, 44]:
        prune_layers.append("conv2d_"+str(i))

global glb_mask
glb_mask = {}

def prune_depthwise(model, dw_sparsity, pw_sparsity, current_epoch):
    global glb_mask
    global prune_layers

    for l_number, layer in enumerate(model.layers):
        if (isinstance(layer, tf.keras.layers.DepthwiseConv2D) and l_number > 11) or (isinstance(layer, tf.keras.layers.Conv2D) and layer.name in prune_layers):
            dw_weights = layer.get_weights()[0]

            if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
                sparsity = dw_sparsity
            else:
                sparsity = pw_sparsity

            if current_epoch < EPOCHS - RECOVERY_EPOCHS:
                #Get abs weights
                abs_weights = tf.math.abs(dw_weights)
                #find number of elements needed for targeted sparsity
                k = tf.dtypes.cast(
                  tf.math.maximum(
                     tf.math.round(
                      tf.dtypes.cast(
                          tf.size(abs_weights), tf.float32) * (1 - sparsity)), 1), tf.int32)
                #Sort the entire array
                values, _ = tf.math.top_k( tf.reshape(abs_weights, [-1]), k=tf.size(abs_weights))
                # Grab the (k-1)th value
                current_threshold = tf.gather(values, k - 1)
                #create mask based on threshold
                mask = tf.dtypes.cast( tf.math.greater_equal(abs_weights, current_threshold), dw_weights.dtype)
                glb_mask[l_number] = mask
            else:
                mask = glb_mask[l_number]
            dw_weights_sparse = tf.math.multiply(dw_weights, mask)
            dw_weights_numpy = dw_weights_sparse.numpy()
            layer.set_weights([dw_weights_numpy])



class PruneDepthwise(tf.keras.callbacks.Callback):
    def __init__(self, dw_schedule, pw_schedule):
        super(PruneDepthwise, self).__init__()
        self.dw_schedule = dw_schedule
        self.pw_schedule = pw_schedule
        self.epoch_number = 0

    def on_train_batch_end(self, batch, logs=None):
        prune_depthwise(self.model, self.dw_schedule[self.epoch_number], self.pw_schedule[self.epoch_number], self.epoch_number)

    def on_epoch_end(self, epoch, logs=None):
        prune_depthwise(self.model, self.dw_schedule[self.epoch_number], self.pw_schedule[self.epoch_number], self.epoch_number)
        self.epoch_number += 1







#****************************************************************************************************
#####################################################################################################
#Main Function
LOSS = ['categorical_crossentropy']
ACCURACY = ['accuracy']


def generate_checkpoint(model_name):
    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_name,
                monitor='val_accuracy', verbose=1, save_best_only=True,\
                save_weights_only=False, mode='auto', save_freq='epoch')
    return checkpoint

#Depthwise pruning schedule
def pruning_schedule(init_sparsity, target_sparsity, pruning_steps, recovery=RECOVERY_EPOCHS, pow=3, dt=100, begin_step=0):

    sparsities = []

    si = init_sparsity
    sf = target_sparsity

    t = []
    to = begin_step
    for n in range(0,pruning_steps):
        t.append(to+n*dt)

    for step in t:
        sparsities.append(sf + (si - sf) * (1 - (step - to) / (pruning_steps * dt))**pow)

    for step in range(recovery):
        sparsities.append(target_sparsity)

    return sparsities



def lr_schedule():
    rates = []
    initial_learning_rate = INIT_LEARNING_RATE
    steps = EPOCHS
    decay_steps = EPOCHS
    end_learning_rate=0.0001
    power = 3.0

    for step in range(steps):
        step = min(step, decay_steps)
        new_lr = ( (initial_learning_rate - end_learning_rate) * (1 - step / decay_steps) ** (power)) + end_learning_rate
        rates.append(new_lr)

    return rates


def get_callbacks():
    learning_rates = lr_schedule()
    MODEL_NAME = pruned_model
    log_dir = log_dir_parent + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint = generate_checkpoint(MODEL_NAME)
    dw_schedule = pruning_schedule(DW_INIT_SPARSITY, DW_FINAL_SPARSITY, EPOCHS-RECOVERY_EPOCHS)
    pw_schedule = pruning_schedule(PW_INIT_SPARSITY, PW_FINAL_SPARSITY, EPOCHS-RECOVERY_EPOCHS)
    callbacks = []
    callbacks.append(checkpoint)
    callbacks.append( PruneDepthwise(dw_schedule, pw_schedule) )
    callbacks.append(tf.keras.callbacks.LearningRateScheduler(lambda epoch: float(learning_rates[epoch])))
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))
    return callbacks



def main():
    global vanilla_layers, DATA_DIR, training, pre_trained, TOTAL_GPUS, ACTIVE_GPUS
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
            model = tf.keras.models.load_model(vanilla_model)
            optimizer = tf.keras.optimizers.SGD(learning_rate=INIT_LEARNING_RATE, momentum=0.9)
            model.compile(loss=LOSS, optimizer=optimizer, metrics=ACCURACY)

        else:
            model = tf.keras.models.load_model(pruned_model)

        if training:
            model.fit(train, validation_data=val, validation_freq=1,
                 steps_per_epoch=int(train_size / BATCH_SIZE), validation_steps=int(val_size / BATCH_SIZE),
                 batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=get_callbacks())

        else:
            #model.summary()
            model.evaluate(val, steps=int(val_size/BATCH_SIZE))
#####################################################################################################
main()
