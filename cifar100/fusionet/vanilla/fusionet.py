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
model_name = "vanilla.h5"

TOTAL_GPUS = 4
ACTIVE_GPUS = 4
GPU_MEM = 9.5e3
CPUs = 8

IMG_SIZE = IMG_H = IMG_W = 32
BATCH_SIZE = 16 * ACTIVE_GPUS
EPOCHS = 140
CLASSES = 100
VERBOSE = 1
VALIDATION_SPLIT = 0.1
WDECAY = 4e-5
INIT_LEARNING_RATE = 0.006

training = False
pre_trained = True

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

#setup_gpus(mem_alloc=GPU_MEM, num_gpu=ACTIVE_GPUS)

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
def ch_shuffle(x, pix, in_ch, groups):
    ch_per_group = in_ch // groups
    x_reshaped = tf.keras.layers.Reshape((pix, pix, groups, ch_per_group))(x)
    x_transposed = tf.keras.layers.Permute((1,2,4,3))(x_reshaped)
    x_shuffled = tf.keras.layers.Reshape((pix, pix, in_ch))(x_transposed)
    return x_shuffled

def group_conv(input, in_ch, out_ch, groups, decay=WDECAY):
    in_group_ch = in_ch // groups
    out_group_ch = out_ch // groups
    tmp_list = []

    for group in range(groups):
        group_index = group * in_group_ch
        tmp_list.append(tf.keras.layers.Conv2D(filters=out_group_ch, kernel_size=(1,1),\
           padding='same', use_bias=False, kernel_initializer="he_normal",\
           kernel_regularizer=regularizers.l2(decay))(input[:,:,:,group_index:group_index+in_group_ch]))

    return tf.keras.layers.Concatenate()(tmp_list)



def ShuffleUnit(input, in_ch, out_ch, groups=4, expansion=4, stride=1, pix=32, decay=WDECAY, residual=True):

    midd_ch = int(in_ch * expansion)
    #************* PW1 Group Conv **************************
    pw1 = group_conv(input, in_ch=in_ch, out_ch=midd_ch, groups=groups, decay=decay)
    pw1_bn = tf.keras.layers.BatchNormalization()(pw1)
    pw1_rl = tf.keras.layers.ReLU()(pw1_bn)

    #*************     DW Conv    **************************
    dw = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), padding='same', strides = stride, use_bias=False, kernel_initializer="he_normal")(pw1_rl)
    dw_bn = tf.keras.layers.BatchNormalization()(dw)
    dw_rl = tf.keras.layers.ReLU()(dw_bn)


    if residual:
        pw2 = group_conv(dw_rl, in_ch=midd_ch, out_ch=out_ch if stride==1 and in_ch==out_ch else out_ch-in_ch, groups=groups, decay=decay)
    else:
        pw2 = group_conv(dw_rl, in_ch=midd_ch, out_ch=out_ch, groups=groups, decay=decay)
    pw2_bn = tf.keras.layers.BatchNormalization()(pw2)
    pw2_shuffled = ch_shuffle(pw2_bn, pix, out_ch, groups)



    if stride == 1 and in_ch == out_ch and residual:
        result = tf.math.add(pw2_shuffled, input)
    elif residual:
        res = tf.keras.layers.AveragePooling2D(pool_size=3, strides=stride, padding='same')(input)
        result = tf.keras.layers.Concatenate()([pw2_shuffled, res])
    else:
        result = pw2_shuffled

    return result



def build_model():
    global WDECAY
    inp = tf.keras.Input(shape=(32,32,3))
    Conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', use_bias=False, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(inp)
    Conv1_bn = tf.keras.layers.BatchNormalization()(Conv1)
    Conv1_rl = tf.keras.layers.ReLU()(Conv1_bn)

    pw_init = group_conv(Conv1_rl, in_ch=32, out_ch=32, groups=4, decay=4e-5)
    pw_init_bn = tf.keras.layers.BatchNormalization()(pw_init)
    pw_init_rl = tf.keras.layers.ReLU()(pw_init_bn)
    pw_init_shuffled = ch_shuffle(pw_init_rl, 32, 32, 4)
    dw_init = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), padding='same', strides = 1, use_bias=False, kernel_initializer="he_normal")(pw_init_shuffled)
    dw_init_bn = tf.keras.layers.BatchNormalization()(dw_init)

    WDECAY = 1e-4
    block1 = ShuffleUnit(input=dw_init_bn, in_ch=32, out_ch=32, pix=32, expansion=74, residual=False)
    block2 = ShuffleUnit(input=block1, in_ch=32, out_ch=32, pix=32, expansion=74)
    block3 = ShuffleUnit(input=block2, in_ch=32, out_ch=32, pix=32, expansion=74)

    block4 = ShuffleUnit(input=block3, in_ch=32, out_ch=64, pix=16, expansion=74, stride=2, residual=False)
    block5 = ShuffleUnit(input=block4, in_ch=64, out_ch=64, pix=16, expansion=37)
    block6 = ShuffleUnit(input=block5, in_ch=64, out_ch=64, pix=16, expansion=37)
    block7 = ShuffleUnit(input=block6, in_ch=64, out_ch=64, pix=16, expansion=37)

    WDECAY = 4e-5
    block8 = ShuffleUnit(input=block7, in_ch=64, out_ch=96, pix=8, expansion=37, stride=2, residual=False)
    block9 = ShuffleUnit(input=block8, in_ch=96, out_ch=96, pix=8, expansion=36)
    block10 = ShuffleUnit(input=block9, in_ch=96, out_ch=96, pix=8, expansion=36)

    block11 = ShuffleUnit(input=block10, in_ch=96, out_ch=96, pix=8, expansion=36)
    block12 = ShuffleUnit(input=block11, in_ch=96, out_ch=96, pix=8, expansion=36)
    block13 = ShuffleUnit(input=block12, in_ch=96, out_ch=96, pix=8, expansion=36)

    block14 = ShuffleUnit(input=block13, in_ch=96, out_ch=96, pix=8, expansion=36)
    block15 = ShuffleUnit(input=block14, in_ch=96, out_ch=96, pix=8, expansion=36)
    block16 = ShuffleUnit(input=block15, in_ch=96, out_ch=160, pix=8, expansion=36, residual=False)

    expanded_lastConv = tf.keras.layers.Conv2D(filters=1280, kernel_size=(1,1), padding='same', use_bias=False, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(WDECAY))(block16)
    #pw_final = group_conv(block16, in_ch=160, out_ch=1280, groups=4)
    pw_final_bn = tf.keras.layers.BatchNormalization()(expanded_lastConv)
    pw_final_rl = tf.keras.layers.ReLU()(pw_final_bn)

    global_ave_pool = tf.keras.layers.GlobalAveragePooling2D()(pw_final_rl)
    dense = tf.keras.layers.Dense(100, activation='softmax', use_bias=True, kernel_initializer = 'glorot_uniform',bias_initializer = 'zeros', kernel_regularizer=regularizers.l2(4e-5))(global_ave_pool)
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
    checkpoint = tf.keras.callbacks.ModelCheckpoint("vanilla2.h5",
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
