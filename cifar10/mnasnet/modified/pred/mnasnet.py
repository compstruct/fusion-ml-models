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
pruned_model = "post_prune.h5"
pred_model = "pred.h5"
pred_model_lastepoch = "pred_lastepoch.h5"

TOTAL_GPUS = 4
ACTIVE_GPUS = 4
GPU_MEM = 10e3
CPUs = 8

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
train_ternary=True
pred1_quantized = True
pred2_quantized = True
vanilla_layers = []
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

TOTAL_GPUS = int(sys.argv[4])
ACTIVE_GPUS = TOTAL_GPUS
setup_gpus(mem_alloc=GPU_MEM, num_gpu=ACTIVE_GPUS)


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

    train = train.repeat().shuffle(shuffle_buff_size).map(preprocess, num_parallel_calls=CPUs).batch(batch_size).map(augmentation, num_parallel_calls=CPUs)
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
def ternary_conv(x, in_ch, out_ch, k=3, stride=1, padding='SAME', wi=0, train_weights=train_ternary):
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
def ternary_dw_conv(x, in_ch, out_ch, k=3, stride=1, padding='SAME', wi=0, train_weights=train_ternary):
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
def MNAS_block(inp, in_ch, midd_ch, out_ch, kernel_size, stride=1, SE=False, SE_ratio=16, decay=WDECAY, i=0):

    global vanilla_layers
    layers = vanilla_layers
    i = i

    decompress = tf.keras.layers.Conv2D(filters=midd_ch, kernel_size=1, padding='same', use_bias=False,\
                      kernel_initializer=tf.constant_initializer(layers[i].get_weights()[0]),\
                      kernel_regularizer=regularizers.l2(decay), trainable=not train_ternary)(inp)
    if pred1_quantized:
        in_quantized = tf.quantization.fake_quant_with_min_max_args(inp, min=-6, max=6, num_bits=4)
    else:
        in_quantized = inp

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
    convdw = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, padding='same', strides = stride, use_bias=False, depthwise_initializer=tf.constant_initializer(layers[i].get_weights()[0]), trainable=(not train_ternary))(decompress_relu)
    if pred2_quantized:
        decompress_quantized = tf.quantization.fake_quant_with_min_max_args(decompress_tr_relu, min=-6, max=6, num_bits=4)
    else:
        decompress_quantized = decompress_tr_relu
    convdw_tr = ternary_dw_conv(decompress_quantized, in_ch, midd_ch, k=kernel_size, stride=stride, wi=i, train_weights=train_ternary)
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



    compress = tf.keras.layers.Conv2D(filters=out_ch, kernel_size=(1,1), padding='same', use_bias=False,\
                          kernel_initializer=tf.constant_initializer(layers[i].get_weights()[0]), kernel_regularizer=regularizers.l2(decay), trainable=not train_ternary)(convdw_relu)
    i+=1
    compress_bn = tf.keras.layers.BatchNormalization(beta_initializer=tf.constant_initializer(layers[i].get_weights()[1]), gamma_initializer=tf.constant_initializer(layers[i].get_weights()[0]),\
                         moving_mean_initializer=tf.constant_initializer(layers[i].get_weights()[2]), moving_variance_initializer=tf.constant_initializer(layers[i].get_weights()[3]))(compress)
    i+=1


    if SE:
        squeeze = tf.keras.layers.GlobalAveragePooling2D()(compress_bn)
        squeeze1 = tf.expand_dims(squeeze, 1)
        squeeze2 = tf.expand_dims(squeeze1, 1)
        i+=3
        excite1 = tf.keras.layers.Conv2D(filters=out_ch*SE_ratio, kernel_size=1, padding='same', use_bias=True,\
                  kernel_initializer=tf.constant_initializer(layers[i].get_weights()[0]),\
                  bias_initializer=tf.constant_initializer(layers[i].get_weights()[1]), kernel_regularizer=regularizers.l2(4e-5))(squeeze2)
        excite1_rl = tf.keras.layers.ReLU()(excite1)
        i+=2
        excite2 = tf.keras.layers.Conv2D(filters=out_ch, kernel_size=1, padding='same', use_bias=True,\
                  kernel_initializer=tf.constant_initializer(layers[i].get_weights()[0]),\
                  bias_initializer=tf.constant_initializer(layers[i].get_weights()[1]),\
                  kernel_regularizer=regularizers.l2(4e-5))(excite1_rl)
        excitation = tf.math.sigmoid(excite2)
        com_excited = tf.math.multiply(excitation, compress_bn)
        i+=3
    else:
        com_excited = compress_bn

    if stride == 1 and in_ch==out_ch:
        res = tf.math.add(com_excited, inp)
        i+=1
    else:
        res = com_excited
    return res, i


#*****************************************************************************************************
def build_model():
    global vanilla_layers
    layers = vanilla_layers
    i = 1
    inp = tf.keras.Input(shape=(32,32,3))
    Conv1 = tf.keras.layers.Conv2D(32, (3,3),padding='same', use_bias=False,\
          kernel_initializer=tf.constant_initializer(layers[i].get_weights()[0]), kernel_regularizer=regularizers.l2(4e-5))(inp)
    i+=1
    bn_Conv1 = tf.keras.layers.BatchNormalization(beta_initializer=tf.constant_initializer(layers[i].get_weights()[1]), gamma_initializer=tf.constant_initializer(layers[i].get_weights()[0]),\
                         moving_mean_initializer=tf.constant_initializer(layers[i].get_weights()[2]), moving_variance_initializer=tf.constant_initializer(layers[i].get_weights()[3]))(Conv1)
    relu_Conv1 = tf.keras.layers.ReLU()(bn_Conv1)
    i+=2

    expanded_conv_depthwise = tf.keras.layers.DepthwiseConv2D((3,3), padding='same', use_bias=False,\
                              depthwise_initializer=tf.constant_initializer(layers[i].get_weights()[0]), kernel_regularizer=regularizers.l2(4e-5))(relu_Conv1)
    i+=1
    expanded_conv_depthwise_BN = tf.keras.layers.BatchNormalization(beta_initializer=tf.constant_initializer(layers[i].get_weights()[1]), gamma_initializer=tf.constant_initializer(layers[i].get_weights()[0]),\
                         moving_mean_initializer=tf.constant_initializer(layers[i].get_weights()[2]), moving_variance_initializer=tf.constant_initializer(layers[i].get_weights()[3]))(expanded_conv_depthwise)
    expanded_conv_depthwise_relu = tf.keras.layers.ReLU()(expanded_conv_depthwise_BN)
    i+=2
    expanded_conv_project = tf.keras.layers.Conv2D(16,(1,1),padding='same', use_bias=False,\
                  kernel_initializer=tf.constant_initializer(layers[i].get_weights()[0]), kernel_regularizer=regularizers.l2(4e-5))(expanded_conv_depthwise_relu)
    i+=1
    b0 = tf.keras.layers.BatchNormalization(beta_initializer=tf.constant_initializer(layers[i].get_weights()[1]), gamma_initializer=tf.constant_initializer(layers[i].get_weights()[0]),\
                         moving_mean_initializer=tf.constant_initializer(layers[i].get_weights()[2]), moving_variance_initializer=tf.constant_initializer(layers[i].get_weights()[3]))(expanded_conv_project)
    i+=1

    b1,i = MNAS_block(b0, 16, 960, 24, kernel_size=3, decay=8e-4, i=i)
    b2,i = MNAS_block(b1, 24, 960, 24, kernel_size=3, decay=8e-4, i=i)

    b3,i = MNAS_block(b2, 24, 960, 24, kernel_size=5, stride=2, SE=True, decay=8e-4, i=i)
    b4,i = MNAS_block(b3, 24, 960, 32, kernel_size=5, SE=True, decay=4e-4, i=i)
    b5,i = MNAS_block(b4, 32, 960, 32, kernel_size=5, SE=True, decay=4e-4, i=i)

    b6,i = MNAS_block(b5, 32, 960, 32, kernel_size=3, decay=4e-4, i=i)
    b7,i = MNAS_block(b6, 32, 960, 32, kernel_size=3, decay=4e-4, i=i)
    b8,i = MNAS_block(b7, 32, 960, 40, kernel_size=3, decay=4e-4, i=i)

    b9,i = MNAS_block(b8, 40, 960, 40, kernel_size=3, stride=2, decay=4e-4, i=i)
    b10,i = MNAS_block(b9, 40, 960, 40, kernel_size=3, SE=True, SE_ratio=64, i=i)
    b11,i = MNAS_block(b10, 40, 960, 40, kernel_size=3, SE=True, SE_ratio=128, i=i)

    b12,i = MNAS_block(b11, 40, 960, 40, kernel_size=5, SE=True, SE_ratio=64, i=i)
    b13,i = MNAS_block(b12, 40, 960, 40, kernel_size=5, SE=True, SE_ratio=128, i=i)
    b14,i = MNAS_block(b13, 40, 960, 40, kernel_size=5, SE=True, SE_ratio=128, i=i)

    b15,i = MNAS_block(b14, 40, 960, 80, kernel_size=3, i=i)

    global_ave_pool = tf.keras.layers.GlobalAveragePooling2D()(b15)
    i+=1
    dense = tf.keras.layers.Dense(10, activation='softmax', use_bias=True,\
                       kernel_initializer = tf.constant_initializer(layers[i].get_weights()[0]), bias_initializer = tf.constant_initializer(layers[i].get_weights()[1]), kernel_regularizer=regularizers.l2(4e-5))(global_ave_pool)

    model = tf.keras.models.Model(inputs=inp, outputs=dense)
    return model



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
    checkpoint = tf.keras.callbacks.ModelCheckpoint(pred_model,
                monitor='val_accuracy', verbose=1, save_best_only=True,\
                save_weights_only=False, mode='auto', save_freq='epoch')
    checkpoint2 = tf.keras.callbacks.ModelCheckpoint(pred_model_lastepoch,
                monitor='val_accuracy', verbose=1, save_best_only=True,\
                save_weights_only=False, mode='auto', save_freq='epoch')
    return checkpoint, checkpoint2

def get_callbacks():
    learning_rates = generate_learning_rates1()
    MODEL_NAME = pred_model
    log_dir = log_dir_parent + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint, checkpoint2 = generate_checkpoint(MODEL_NAME)
    callbacks = []
    callbacks.append(checkpoint)
    callbacks.append(checkpoint2)
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

    with tf.device('/cpu:0'):
        train, info_train, val, info_val, train_size, val_size = get_dataset(DATASET, shuffle_buff_size=25*1024)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        if not pre_trained:
            vanilla_layers = tf.keras.models.load_model(pruned_model).layers
            model = build_model()
            optimizer = tf.keras.optimizers.SGD(learning_rate=INIT_LEARNING_RATE, momentum=0.9)
            model.compile(loss=LOSS, optimizer=optimizer, metrics=ACCURACY)
        else:
            vanilla_layers = tf.keras.models.load_model(pruned_model).layers
            model = build_model()
            model.load_weights(pred_model_lastepoch, by_name=True)
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
            model.evaluate(val, steps=int(val_size/BATCH_SIZE))
            #model.summary()
#####################################################################################################

main()
