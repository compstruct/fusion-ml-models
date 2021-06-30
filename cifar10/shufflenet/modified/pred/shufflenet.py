#!/usr/bin/env python
# coding: utf-8

#####################################################################################################
import sys
sys.path = ['', '/usr/lib/python36.zip', '/usr/lib/python3.6', '/usr/lib/python3.6/lib-dynload',
            '/home/mohamadol/.local/lib/python3.6/site-packages', '/usr/local/lib/python3.6/dist-packages']

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
pruned_model = "post_prune.h5"
model_name = "pred.h5"

TOTAL_GPUS = 2
ACTIVE_GPUS = 2
IMG_SIZE = IMG_H = IMG_W = 32
BATCH_SIZE = 12 * ACTIVE_GPUS
EPOCHS = 30
CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.1
WDECAY = 1e-4
INIT_LEARNING_RATE = 1e-3

training = False
train_round2 = False
model_summary = False
pre_trained = True
train_ternary = True
vanilla_layers = []
pred_fmaps_qtz = True


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
#****************************Ternary Stuff***********************************************************
def compute_threshold(x):
    x_sum = tf.stop_gradient(tf.reduce_sum(tf.abs(x), keepdims =False))
    size = tf.stop_gradient(tf.cast(tf.size(x), tf.float32))
    threshold = tf.stop_gradient(tf.math.scalar_mul(0.7/size, x_sum))
    return threshold

#Compute constant alpha
def compute_alpha(x):
    threshold = compute_threshold(x)
    #Weights above delta
    alpha1_temp1 = tf.stop_gradient(tf.where(tf.greater(x, threshold), x, tf.zeros_like(x, tf.float32)))
    #Weights below negative delta
    alpha1_temp2 = tf.stop_gradient(tf.where(tf.less(x,-threshold), x, tf.zeros_like(x, tf.float32)))
    #replace all the non-zero elements with 1
    alpha_array = tf.stop_gradient(tf.math.add(alpha1_temp1, alpha1_temp2))
    alpha_array_abs = tf.stop_gradient(tf.math.abs(alpha_array))
    alpha_array_abs1 = tf.stop_gradient(tf.where(tf.greater(alpha_array_abs,0),tf.ones_like(alpha_array_abs,tf.float32), tf.zeros_like(alpha_array_abs, tf.float32)))
    #divide actual sum of elements with size of non-zeros
    alpha_sum = tf.stop_gradient(tf.reduce_sum(alpha_array_abs))
    n = tf.stop_gradient(tf.reduce_sum(alpha_array_abs1))
    alpha = tf.stop_gradient(tf.math.divide(alpha_sum,n))
    return alpha

#Here we ternerize weight based on threshold to -1, 0 or 1
def ternerize(x):
    g = tf.compat.v1.get_default_graph()
    #with ops.name_scope("tenarized") as name:
    with g.gradient_override_map({"Sign": "Identity"}):
        threshold =compute_threshold(x)
        x=tf.sign(tf.add(tf.sign(tf.add(x,threshold)),tf.sign(tf.add(x,-threshold))))
        return x

#Here we perform the convolution using ternary weights
def ternary_conv(x, in_ch, out_ch, k=3, stride=1, padding='SAME', wi=0, train_weights=True):
    global vanilla_layers
    layers = vanilla_layers
    initializer = tf.constant_initializer(layers[wi].get_weights()[0])
    w = tf.Variable(initializer(shape=(k, k, in_ch, out_ch)), trainable=train_weights)
    alpha_w = compute_alpha(w)
    ter_w = ternerize(w)
    ter_w_alpha = tf.multiply(alpha_w, ter_w)
    out = tf.nn.conv2d(x, ter_w_alpha, strides=[1, stride, stride, 1], padding=padding)
    return out

#Here we perform the depthwise-convolution using ternary weights
def ternary_dw_conv(x, in_ch, out_ch, k=3, stride=1, padding='SAME', wi=0, train_weights=True):
    global vanilla_layers
    layers = vanilla_layers
    initializer = tf.constant_initializer(layers[wi].get_weights()[0])
    w = tf.Variable(initializer(shape=(k, k, out_ch, 1)), trainable=train_weights)
    alpha_w = compute_alpha(w)
    ter_w = ternerize(w)
    ter_w_alpha = tf.multiply(alpha_w, ter_w)
    out = tf.nn.depthwise_conv2d(x, ter_w_alpha, strides=[1, stride, stride, 1], padding=padding)
    return out



#####################################################################################################
def ch_shuffle(x, pix, in_ch, groups):
    ch_per_group = in_ch // groups
    x_reshaped = tf.keras.layers.Reshape((pix, pix, groups, ch_per_group))(x)
    x_transposed = tf.keras.layers.Permute((1,2,4,3))(x_reshaped)
    x_shuffled = tf.keras.layers.Reshape((pix, pix, in_ch))(x_transposed)
    return x_shuffled


def group_conv(input, in_ch, out_ch, groups, i, decay=WDECAY, quantized=False):
    global vanilla_layers
    layers = vanilla_layers

    in_group_ch = in_ch // groups
    out_group_ch = out_ch // groups
    tmp_list = []

    for group in range(groups):
        group_index = group * in_group_ch
        if not quantized:
            tmp_list.append(tf.keras.layers.Conv2D(filters=out_group_ch, kernel_size=(1,1),\
                padding='same', use_bias=False, kernel_initializer=tf.constant_initializer(layers[i+3+group].get_weights()[0]),\
                kernel_regularizer=regularizers.l2(decay), trainable=not train_ternary)(input[:,:,:,group_index:group_index+in_group_ch]))
        else:
            tmp_list.append(ternary_conv(input[:,:,:,group_index:group_index+in_group_ch],\
                            in_group_ch, out_group_ch, k=1, wi=i+3+group, train_weights=train_ternary))


    return tf.keras.layers.Concatenate()(tmp_list)



#************************************************************************************************************************
#**********************************************Shuffle Net Block*********************************************************
def ShuffleUnit(input, in_ch, out_ch, groups=3, block=0, expansion=4,\
                stride=1, pix=32, decay=WDECAY, residual=True, i=0):

    global vanilla_layers, pred_fmaps_qtz_bits
    layers = vanilla_layers
    midd_ch = int(in_ch * expansion)

    #********************************************************
    #***************Firrst PointWise*************************
    #Group Conv
    pw2    = group_conv(input, in_ch=in_ch, out_ch=midd_ch, groups=groups, i=i, decay=decay)
    #Quantization for PW1 prediction
    if pred_fmaps_qtz:
        pw1_quantized = tf.quantization.fake_quant_with_min_max_args(input, min=-6, max=6, num_bits=4)
    else:
        pw1_quantized = input
    pw2_tr = group_conv(pw1_quantized, in_ch=in_ch, out_ch=midd_ch, groups=groups, i=i, quantized=True)
    i+=7
    #Batch Norm
    pw2_bn = tf.keras.layers.BatchNormalization(beta_initializer=tf.constant_initializer(layers[i].get_weights()[1]),\
                                                   gamma_initializer=tf.constant_initializer(layers[i].get_weights()[0]),\
                                                   moving_mean_initializer=tf.constant_initializer(layers[i].get_weights()[2]),\
                                moving_variance_initializer=tf.constant_initializer(layers[i].get_weights()[3]), trainable = not train_ternary)(pw2)
    pw2_tr_bn = tf.keras.layers.BatchNormalization(beta_initializer=tf.constant_initializer(layers[i].get_weights()[1]),\
                                                  gamma_initializer=tf.constant_initializer(layers[i].get_weights()[0]),\
                                                  moving_mean_initializer=tf.constant_initializer(layers[i].get_weights()[2]),\
                                 moving_variance_initializer=tf.constant_initializer(layers[i].get_weights()[3]), trainable = train_ternary)(pw2_tr)
    #ReLU
    pw2_tr_relu = tf.keras.layers.ReLU()(pw2_tr_bn)
    approx_grad = tf.cond(train_ternary, lambda: tf.identity(pw2_tr_relu), lambda: tf.keras.layers.ReLU()(pw2_bn))
    pw2_rl = approx_grad + tf.stop_gradient(tf.where(tf.math.greater(pw2_tr_relu, 0.), pw2_bn, tf.zeros_like(pw2_bn)) - approx_grad)
    #pw2_rl = tf.keras.layers.ReLU()(pw2_bn)
    #Channel Shuffle
    pw2_shuffled = ch_shuffle(pw2_rl, pix, midd_ch, groups)
    pw2_tr_shuffled = ch_shuffle(pw2_tr_relu, pix, midd_ch, groups)
    i+=5
    #*********************************************************

    #********************************************************
    #***************Second PointWise*************************
    #Quantize fmaps
    if pred_fmaps_qtz:
        pw2_quantized = tf.quantization.fake_quant_with_min_max_args(pw2_tr_shuffled, min=-6, max=6, num_bits=4)
    else:
        pw2_quantized = pw2_tr_shuffled
    #Group Conv
    if residual:
        pw1 = group_conv(pw2_shuffled, in_ch=midd_ch, out_ch=out_ch if stride==1 and in_ch==out_ch else out_ch-in_ch, groups=groups, i=i, decay=decay)
        pw1_tr = group_conv(pw2_quantized, in_ch=midd_ch, out_ch=out_ch if stride==1 and in_ch==out_ch else out_ch-in_ch, groups=groups, i=i, quantized=True)
    else:
        pw1 = group_conv(pw2_shuffled, in_ch=midd_ch, out_ch=out_ch, groups=groups, i=i, decay=decay)
        pw1_tr = group_conv(pw2_quantized, in_ch=midd_ch, out_ch=out_ch, groups=groups, i=i, quantized=True)
    i+=7
    #Batch Norm
    pw1_bn = tf.keras.layers.BatchNormalization(beta_initializer=tf.constant_initializer(layers[i].get_weights()[1]), gamma_initializer=tf.constant_initializer(layers[i].get_weights()[0]),\
                         moving_mean_initializer=tf.constant_initializer(layers[i].get_weights()[2]), moving_variance_initializer=tf.constant_initializer(layers[i].get_weights()[3]), trainable = not train_ternary)(pw1)
    pw1_tr_bn = tf.keras.layers.BatchNormalization(beta_initializer=tf.constant_initializer(layers[i].get_weights()[1]), gamma_initializer=tf.constant_initializer(layers[i].get_weights()[0]),\
                         moving_mean_initializer=tf.constant_initializer(layers[i].get_weights()[2]), moving_variance_initializer=tf.constant_initializer(layers[i].get_weights()[3]), trainable = train_ternary)(pw1_tr)
    #ReLU
    pw1_tr_relu = tf.keras.layers.ReLU()(pw1_tr_bn)
    approx_grad = tf.cond(train_ternary, lambda: tf.identity(pw1_tr_relu), lambda: tf.keras.layers.ReLU()(pw1_bn))
    pw1_rl = approx_grad + tf.stop_gradient(tf.where(tf.math.greater(pw1_tr_relu, 0.), pw1_bn, tf.zeros_like(pw1_bn)) - approx_grad)
    #pw1_rl = tf.keras.layers.ReLU()(pw1_bn)
    i+=2
    #*********************************************************

    #****************Depthwise Convolution*******************
    dw = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), padding='same', strides = stride, use_bias=False, depthwise_initializer=tf.constant_initializer(layers[i].get_weights()[0]))(pw1_rl)
    i+=1
    dw_bn = tf.keras.layers.BatchNormalization(beta_initializer=tf.constant_initializer(layers[i].get_weights()[1]), gamma_initializer=tf.constant_initializer(layers[i].get_weights()[0]),\
                         moving_mean_initializer=tf.constant_initializer(layers[i].get_weights()[2]), moving_variance_initializer=tf.constant_initializer(layers[i].get_weights()[3]))(dw)

    if stride == 1 and in_ch == out_ch and residual:
        result = tf.math.add(dw_bn, input)
        i+=2
    elif residual:
        res = tf.keras.layers.AveragePooling2D(pool_size=3, strides=stride, padding='same')(input)
        result = tf.keras.layers.Concatenate()([dw_bn, res])
        i+=3
    else:
        result = dw_bn
        i+=1

    return result, i






def build_model():
    global vanilla_layers
    layers = vanilla_layers
    i = 1

    inp = tf.keras.Input(shape=(32,32,3))
    Conv1 = tf.keras.layers.Conv2D(filters=24, kernel_size=(3,3), padding='same', use_bias=False, kernel_initializer=tf.constant_initializer(layers[i].get_weights()[0]), kernel_regularizer=regularizers.l2(4e-5))(inp)
    i+=1
    Conv1_bn = tf.keras.layers.BatchNormalization(beta_initializer=tf.constant_initializer(layers[i].get_weights()[1]), gamma_initializer=tf.constant_initializer(layers[i].get_weights()[0]),\
                         moving_mean_initializer=tf.constant_initializer(layers[i].get_weights()[2]), moving_variance_initializer=tf.constant_initializer(layers[i].get_weights()[3]))(Conv1)
    Conv1_rl = tf.keras.layers.ReLU()(Conv1_bn)
    i+=2

    pw_init = group_conv(Conv1_rl, in_ch=24, out_ch=24, groups=3, i=i, decay=4e-5)
    i+=7
    pw_init_bn = tf.keras.layers.BatchNormalization(beta_initializer=tf.constant_initializer(layers[i].get_weights()[1]), gamma_initializer=tf.constant_initializer(layers[i].get_weights()[0]),\
                         moving_mean_initializer=tf.constant_initializer(layers[i].get_weights()[2]), moving_variance_initializer=tf.constant_initializer(layers[i].get_weights()[3]))(pw_init)
    pw_init_rl = tf.keras.layers.ReLU()(pw_init_bn)
    pw_init_shuffled = ch_shuffle(pw_init_rl, 32, 24, 3)
    i+=5
    dw_init = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), padding='same', strides = 1, use_bias=False, depthwise_initializer=tf.constant_initializer(layers[i].get_weights()[0]))(pw_init_shuffled)
    i+=1
    dw_init_bn = tf.keras.layers.BatchNormalization(beta_initializer=tf.constant_initializer(layers[i].get_weights()[1]), gamma_initializer=tf.constant_initializer(layers[i].get_weights()[0]),\
                         moving_mean_initializer=tf.constant_initializer(layers[i].get_weights()[2]), moving_variance_initializer=tf.constant_initializer(layers[i].get_weights()[3]))(dw_init)
    i+=1

    block1, i = ShuffleUnit(input=dw_init_bn, in_ch=24, out_ch=18, pix=32, expansion=107.25, decay=2e-4, residual=False, i=i, block=1)
    block2, i = ShuffleUnit(input=block1, in_ch=18, out_ch=18, pix=32, expansion=143, decay=2e-4, i=i, block=2)

    block3, i = ShuffleUnit(input=block2, in_ch=18, out_ch=24, pix=32, expansion=143, stride=2, i=i, block=3)
    block4, i = ShuffleUnit(input=block3, in_ch=24, out_ch=24, pix=16, expansion=107.25, i=i, block=4)
    block5, i = ShuffleUnit(input=block4, in_ch=24, out_ch=24, pix=16, expansion=107.25, i=i, block=4)
    block6, i = ShuffleUnit(input=block5, in_ch=24, out_ch=24, pix=16, expansion=107.25, i=i, block=5)

    block7, i = ShuffleUnit(input=block6, in_ch=24, out_ch=42, pix=16, expansion=107.25, stride=2, i=i, block=6)
    block8, i = ShuffleUnit(input=block7, in_ch=42, out_ch=42, pix=8, expansion=62, i=i, block=7)
    block9, i = ShuffleUnit(input=block8, in_ch=42, out_ch=42, pix=8, expansion=62, i=i, block=8)

    block10, i = ShuffleUnit(input=block9, in_ch=42, out_ch=42, pix=8, expansion=62, i=i, block=9)
    block11, i = ShuffleUnit(input=block10, in_ch=42, out_ch=42, pix=8, expansion=62, i=i, block=10)
    block12, i = ShuffleUnit(input=block11, in_ch=42, out_ch=42, pix=8, expansion=62, i=i, block=11)

    block13, i = ShuffleUnit(input=block12, in_ch=42, out_ch=42, pix=8, expansion=62, i=i, block=12)
    block14, i = ShuffleUnit(input=block13, in_ch=42, out_ch=42, pix=8, expansion=62, i=i, block=13)
    block15, i = ShuffleUnit(input=block14, in_ch=42, out_ch=84, pix=8, expansion=62, i=i, block=14)



    pw_final = group_conv(block15, in_ch=84, out_ch=960, groups=3, i=i)
    i+=7
    pw_final_bn = tf.keras.layers.BatchNormalization(beta_initializer=tf.constant_initializer(layers[i].get_weights()[1]), gamma_initializer=tf.constant_initializer(layers[i].get_weights()[0]),\
                         moving_mean_initializer=tf.constant_initializer(layers[i].get_weights()[2]), moving_variance_initializer=tf.constant_initializer(layers[i].get_weights()[3]))(pw_final)
    pw_final_rl = tf.keras.layers.ReLU()(pw_final_bn)


    global_ave_pool = tf.keras.layers.GlobalAveragePooling2D()(pw_final_rl)
    i+=3
    dense = tf.keras.layers.Dense(10, activation='softmax', use_bias=True, kernel_initializer=tf.constant_initializer(layers[i].get_weights()[0]), bias_initializer=tf.constant_initializer(layers[i].get_weights()[1]), kernel_regularizer=regularizers.l2(4e-5))(global_ave_pool)
    return tf.keras.models.Model(inputs=inp, outputs=dense)

#####################################################################################################
#Learing rate & checkpoints
def generate_learning_rates():
    learning_rates=[]
    lr = INIT_LEARNING_RATE
    for i in range(EPOCHS):
        learning_rates.append(lr)
        if i == 7:
            lr = 8e-4
        elif i== 17:
            lr = 4e-4
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

    global vanilla_layers

    with tf.device('/cpu:0'):
        train, info_train, val, info_val, train_size, val_size = get_dataset(DATASET, shuffle_buff_size=25*1024)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        if not pre_trained:
            vanilla_layers = tf.keras.models.load_model(pruned_model).layers
            model = build_model()
            if train_round2:
                model.load_weights(model_name, by_name=True)
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
            if model_summary:
                model.summary()
#####################################################################################################

main()
