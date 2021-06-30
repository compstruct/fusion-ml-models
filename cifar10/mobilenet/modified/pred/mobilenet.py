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
#import tensorflow_model_optimization as tfmot
tfds.disable_progress_bar()
#####################################################################################################



#####################################################################################################
#*****************Global Variables & setup*****************
DATASET = 'cifar10'
DATA_DIR = "/home/mohamadol/tensorflow_datasets"
log_dir_parent = "tb/"
pruned_model = "./../pruned/post_prune.h5"
pred_model = "pred.ckpt"



TOTAL_GPUS = 2
ACTIVE_GPUS = 2
IMG_SIZE = IMG_H = IMG_W = 32
BATCH_SIZE = 32*ACTIVE_GPUS
EPOCHS = 50
CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.1
WDECAY = 1e-4
INIT_LEARNING_RATE = 1e-3

training = False
pre_trained = True
train_ternary = True
pred1_quantized = True
pred2_quantized = True
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

setup_gpus(mem_alloc=11e3, num_gpu=ACTIVE_GPUS)

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

######################################################################################

#####################################################################################################
def InvertedBottleneck(input, in_ch, out_ch, expansion=6, stride=1, l2=WDECAY, layer_index=0):
    global vanilla_layers
    layers = vanilla_layers
    i = layer_index
    midd_ch = int(in_ch * expansion)

    #************************* Decompression*****************************
    decompress = tf.keras.layers.Conv2D(filters=midd_ch, kernel_size=(1,1), padding='same', use_bias=False, kernel_initializer=tf.constant_initializer(layers[i].get_weights()[0]), kernel_regularizer=regularizers.l2(l2), trainable=(not train_ternary))(input)

    if pred1_quantized:
        in_quantized = tf.quantization.fake_quant_with_min_max_args(input, min=-6, max=6, num_bits=4)
    else:
        in_quantized = input

    decompress_tr = ternary_conv(in_quantized, in_ch, midd_ch, 1, wi=i, train_weights=train_ternary)
    i+=1
    decompress_bn = tf.keras.layers.BatchNormalization(beta_initializer=tf.constant_initializer(layers[i].get_weights()[1]), gamma_initializer=tf.constant_initializer(layers[i].get_weights()[0]),\
                         moving_mean_initializer=tf.constant_initializer(layers[i].get_weights()[2]), moving_variance_initializer=tf.constant_initializer(layers[i].get_weights()[3]), trainable=(not train_ternary))(decompress)
    decompress_tr_bn = tf.keras.layers.BatchNormalization(beta_initializer=tf.constant_initializer(layers[i].get_weights()[1]), gamma_initializer=tf.constant_initializer(layers[i].get_weights()[0]),\
                         moving_mean_initializer=tf.constant_initializer(layers[i].get_weights()[2]), moving_variance_initializer=tf.constant_initializer(layers[i].get_weights()[3]), trainable=train_ternary)(decompress_tr)

    decompress_tr_relu = tf.keras.layers.ReLU()(decompress_tr_bn)

    approx_grad = tf.cond(train_ternary, lambda: tf.identity(decompress_tr_relu), lambda: tf.keras.layers.ReLU()(decompress_bn))
    decompress_relu = approx_grad + tf.stop_gradient(tf.where(tf.math.greater(decompress_tr_relu, 0.), decompress_bn, tf.zeros_like(decompress_bn)) - approx_grad)
    i+=2
    #*********************************************************************

    #**************************** Depthwise Convolution *******************
    convdw = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), padding='same', strides = stride, use_bias=False, depthwise_initializer=tf.constant_initializer(layers[i].get_weights()[0]), trainable=(not train_ternary))(decompress_relu)
    if pred2_quantized:
        decompress_quantized = tf.quantization.fake_quant_with_min_max_args(decompress_tr_relu, min=-6, max=6, num_bits=4)
    else:
        decompress_quantized = decompress_tr_relu
    convdw_tr = ternary_dw_conv(decompress_quantized, in_ch, midd_ch, stride=stride, wi=i, train_weights=train_ternary)
    i+=1
    convdw_bn = tf.keras.layers.BatchNormalization(beta_initializer=tf.constant_initializer(layers[i].get_weights()[1]), gamma_initializer=tf.constant_initializer(layers[i].get_weights()[0]),\
                         moving_mean_initializer=tf.constant_initializer(layers[i].get_weights()[2]), moving_variance_initializer=tf.constant_initializer(layers[i].get_weights()[3]), trainable=(not train_ternary))(convdw)
    convdw_tr_bn = tf.keras.layers.BatchNormalization(beta_initializer=tf.constant_initializer(layers[i].get_weights()[1]), gamma_initializer=tf.constant_initializer(layers[i].get_weights()[0]),\
                         moving_mean_initializer=tf.constant_initializer(layers[i].get_weights()[2]), moving_variance_initializer=tf.constant_initializer(layers[i].get_weights()[3]), trainable=train_ternary)(convdw_tr)
    convdw_tr_relu = tf.keras.layers.ReLU()(convdw_tr_bn)
    approx_grad2 = tf.cond(train_ternary, lambda: tf.identity(convdw_tr_relu), lambda: tf.keras.layers.ReLU()(convdw_bn))
    convdw_relu = approx_grad2 + tf.stop_gradient(tf.where(tf.math.greater(convdw_tr_relu, 0.), convdw_bn, tf.zeros_like(convdw_bn)) - approx_grad2)
    i+=2
    #***********************************************************************

    compress = tf.keras.layers.Conv2D(filters=out_ch, kernel_size=(1,1), padding='same', use_bias=False, kernel_initializer=tf.constant_initializer(layers[i].get_weights()[0]), kernel_regularizer=regularizers.l2(l2), trainable=(not train_ternary))(convdw_relu)
    i+=1
    compress_bn = tf.keras.layers.BatchNormalization(beta_initializer=tf.constant_initializer(layers[i].get_weights()[1]), gamma_initializer=tf.constant_initializer(layers[i].get_weights()[0]),\
                         moving_mean_initializer=tf.constant_initializer(layers[i].get_weights()[2]), moving_variance_initializer=tf.constant_initializer(layers[i].get_weights()[3]), trainable=(train_ternary))(compress)

    if in_ch == out_ch and stride == 1:
        result = tf.math.add(compress_bn, input)
        i+=2
    else:
        result = compress_bn
        i+=1
    return result, i


def InvertedBottleneckOrig(input, in_ch, out_ch, expansion=6, stride=1, l2=WDECAY, layer_index=0):
    global vanilla_layers
    layers = vanilla_layers

    i = layer_index
    midd_ch = int(in_ch * expansion)

    decompress = tf.keras.layers.Conv2D(filters=midd_ch, kernel_size=(1,1), padding='same', use_bias=False, kernel_initializer=tf.constant_initializer(layers[i].get_weights()[0]), kernel_regularizer=regularizers.l2(l2), trainable=(train_ternary))(input)
    i+=1
    decompress_bn = tf.keras.layers.BatchNormalization(beta_initializer=tf.constant_initializer(layers[i].get_weights()[1]), gamma_initializer=tf.constant_initializer(layers[i].get_weights()[0]),\
                         moving_mean_initializer=tf.constant_initializer(layers[i].get_weights()[2]), moving_variance_initializer=tf.constant_initializer(layers[i].get_weights()[3]), trainable=(train_ternary))(decompress)
    decompress_relu = tf.keras.layers.ReLU()(decompress_bn)
    i+=2

    convdw = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), padding='same', strides = stride, use_bias=False, depthwise_initializer=tf.constant_initializer(layers[i].get_weights()[0]), trainable=(train_ternary))(decompress_relu)
    i+=1
    convdw_bn = tf.keras.layers.BatchNormalization(beta_initializer=tf.constant_initializer(layers[i].get_weights()[1]), gamma_initializer=tf.constant_initializer(layers[i].get_weights()[0]),\
                         moving_mean_initializer=tf.constant_initializer(layers[i].get_weights()[2]), moving_variance_initializer=tf.constant_initializer(layers[i].get_weights()[3]), trainable=(train_ternary))(convdw)
    convdw_relu = tf.keras.layers.ReLU()(convdw_bn)
    i+=2

    compress = tf.keras.layers.Conv2D(filters=out_ch, kernel_size=(1,1), padding='same', use_bias=False, kernel_initializer=tf.constant_initializer(layers[i].get_weights()[0]), kernel_regularizer=regularizers.l2(l2), trainable=(train_ternary))(convdw_relu)
    i+=1
    compress_bn = tf.keras.layers.BatchNormalization(beta_initializer=tf.constant_initializer(layers[i].get_weights()[1]), gamma_initializer=tf.constant_initializer(layers[i].get_weights()[0]),\
                         moving_mean_initializer=tf.constant_initializer(layers[i].get_weights()[2]), moving_variance_initializer=tf.constant_initializer(layers[i].get_weights()[3]), trainable=(train_ternary))(compress)

    if in_ch == out_ch and stride == 1:
        result = tf.math.add(compress_bn, input)
        i+=2
    else:
        result = compress_bn
        i+=1
    return result, i




def build_model():
    global vanilla_layers
    layers = vanilla_layers

    inp = tf.keras.Input(shape=(32,32,3))
    i = 1
    Conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', use_bias=False, kernel_initializer=tf.constant_initializer(layers[i].get_weights()[0]), kernel_regularizer=regularizers.l2(WDECAY))(inp)
    i+=1
    Conv1_bn = tf.keras.layers.BatchNormalization(beta_initializer=tf.constant_initializer(layers[i].get_weights()[1]), gamma_initializer=tf.constant_initializer(layers[i].get_weights()[0]),\
                         moving_mean_initializer=tf.constant_initializer(layers[i].get_weights()[2]), moving_variance_initializer=tf.constant_initializer(layers[i].get_weights()[3]))(Conv1)
    Conv1_rl = tf.keras.layers.ReLU()(Conv1_bn)
    i+=2

    block0 , i= InvertedBottleneckOrig(input=Conv1_rl, in_ch=32, out_ch=16, expansion=1, layer_index=i)

    block1 , i= InvertedBottleneck(input=block0, in_ch=16, out_ch=24, expansion=(864//16), l2=2e-4, layer_index=i)
    block2 , i= InvertedBottleneck(input=block1, in_ch=24, out_ch=24, expansion=(864//24), l2=2e-4, layer_index=i)
    block3 , i= InvertedBottleneck(input=block2, in_ch=24, out_ch=24, expansion=(864//24), l2=2e-4, layer_index=i)

    block4 , i= InvertedBottleneck(input=block3, in_ch=24, out_ch=32, expansion=(864//24), stride=2, l2=1e-4, layer_index=i)
    block5 , i= InvertedBottleneck(input=block4, in_ch=32, out_ch=32, expansion=(864//32), l2=1e-4, layer_index=i)
    block6 , i= InvertedBottleneck(input=block5, in_ch=32, out_ch=32, expansion=(864//32), l2=1e-4, layer_index=i)
    block7 , i= InvertedBottleneck(input=block6, in_ch=32, out_ch=32, expansion=(864//32), l2=1e-4, layer_index=i)

    block8 , i= InvertedBottleneck(input=block7, in_ch=32, out_ch=40, expansion=(864//32), stride=2, l2=1e-4, layer_index=i)
    block9 , i= InvertedBottleneck(input=block8, in_ch=40, out_ch=40, expansion=(864/40), l2=1e-4, layer_index=i)
    block10 , i= InvertedBottleneck(input=block9, in_ch=40, out_ch=40, expansion=(864/40), l2=1e-4, layer_index=i)
    block11 , i= InvertedBottleneck(input=block10, in_ch=40, out_ch=40, expansion=(864/40), l2=1e-4, layer_index=i)

    block12 , i= InvertedBottleneck(input=block11, in_ch=40, out_ch=40, expansion=(864/40), l2=1e-4, layer_index=i)
    block13 , i= InvertedBottleneck(input=block12, in_ch=40, out_ch=40, expansion=(864/40), l2=1e-4, layer_index=i)
    block14 , i= InvertedBottleneck(input=block13, in_ch=40, out_ch=80, expansion=(864/40), l2=1e-4, layer_index=i)


    expanded_lastConv = tf.keras.layers.Conv2D(filters=1280, kernel_size=(1,1), padding='same', use_bias=False, kernel_initializer=tf.constant_initializer(layers[i].get_weights()[0]), kernel_regularizer=regularizers.l2(WDECAY))(block14)
    i+=1
    expanded_lastConv_bn = tf.keras.layers.BatchNormalization(beta_initializer=tf.constant_initializer(layers[i].get_weights()[1]), gamma_initializer=tf.constant_initializer(layers[i].get_weights()[0]),\
                         moving_mean_initializer=tf.constant_initializer(layers[i].get_weights()[2]), moving_variance_initializer=tf.constant_initializer(layers[i].get_weights()[3]))(expanded_lastConv)
    expanded_lastConv_rl = tf.keras.layers.ReLU()(expanded_lastConv_bn)
    global_ave_pool = tf.keras.layers.GlobalAveragePooling2D()(expanded_lastConv_rl)
    i+=3

    dense = tf.keras.layers.Dense(10, activation='softmax', use_bias=True, kernel_initializer=tf.constant_initializer(layers[i].get_weights()[0]), bias_initializer=tf.constant_initializer(layers[i].get_weights()[1]), kernel_regularizer=regularizers.l2(WDECAY))(global_ave_pool)
    return tf.keras.models.Model(inputs=inp, outputs=dense)

#####################################################################################################
#Learing rate & checkpoints
def generate_learning_rates1():
    learning_rates=[]
    lr = 0.001
    decay_steps = 42
    alpha = 0.001
    total_epochs = 50
    initial_learning_rate = 0.001
    end_learning_rate=0.00008

    steps = total_epochs
    power = 3.0
    for step in range(steps):
        stepp = min(step, decay_steps)
        lr = ( (initial_learning_rate - end_learning_rate) * (1 - stepp / decay_steps) ** (power)) + end_learning_rate
        learning_rates.append(lr)

    return learning_rates

def generate_checkpoint(model_name):
    checkpoint = tf.keras.callbacks.ModelCheckpoint("pred.ckpt",
                monitor='val_accuracy', verbose=1, save_best_only=True,\
                save_weights_only=True, mode='auto', save_freq='epoch')
    return checkpoint

def get_callbacks():
    learning_rates = generate_learning_rates1()
    MODEL_NAME = pred_model
    log_dir = log_dir_parent + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint = generate_checkpoint(MODEL_NAME)
    callbacks = []
    callbacks.append(checkpoint)
    #callbacks.append(checkpoint2)
    callbacks.append(tf.keras.callbacks.LearningRateScheduler(lambda epoch: float(learning_rates[epoch])))
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))
    return callbacks
#####################################################################################################


#####################################################################################################
#Main Function
LOSS = ['categorical_crossentropy']
ACCURACY = ['accuracy']

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
            vanilla_layers = tf.keras.models.load_model(pruned_model).layers
            model = build_model()
            optimizer = tf.keras.optimizers.SGD(learning_rate=INIT_LEARNING_RATE, momentum=0.65)
            model.compile(loss=LOSS, optimizer=optimizer, metrics=ACCURACY)
        else:
            vanilla_layers = tf.keras.models.load_model(pruned_model).layers
            model = build_model()
            model.load_weights(pred_model, by_name=False).expect_partial()
            optimizer = tf.keras.optimizers.SGD(learning_rate=INIT_LEARNING_RATE, momentum=0.9)
            model.compile(loss=LOSS, optimizer=optimizer, metrics=ACCURACY)

        if training:
            model.fit(train,
                epochs=EPOCHS,
                steps_per_epoch=int(train_size / BATCH_SIZE),
                validation_data=val,
                validation_steps=int(val_size / BATCH_SIZE),
                validation_freq=1,
                callbacks=get_callbacks())
        else:
            #model.summary()
            model.evaluate(val, steps=int(val_size/BATCH_SIZE))
#####################################################################################################

main()
