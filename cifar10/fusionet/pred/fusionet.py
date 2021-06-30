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
pruned_model = "post_prune.h5"
pred_model = "pred2.h5"
pred_model_lastepoch = "pred.h5"

#was trained using 4 GPUs
TOTAL_GPUS = 2 #4
ACTIVE_GPUS = 2 #4
GPU_MEM = 10e3
CPUs = 8

IMG_SIZE = IMG_H = IMG_W = 32
BATCH_SIZE = 16 * ACTIVE_GPUS
EPOCHS = 50
CLASSES = 10
VERBOSE = 1
WDECAY = 1e-4
INIT_LEARNING_RATE = 1e-3

training =False
pre_trained = True
model_summary = False
train_ternary = True
vanilla_layers = []
pred_fmaps_qtz = True
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

def group_conv(input, in_ch, out_ch, groups, i, decay=WDECAY, quantized=False, trainable=not train_ternary):
    global vanilla_layers
    layers = vanilla_layers
    in_group_ch = in_ch // groups
    out_group_ch = out_ch // groups
    tmp_list = []

    for group in range(groups):
        group_index = group * in_group_ch
        if not quantized:
            tmp_list.append(tf.keras.layers.Conv2D(filters=out_group_ch, kernel_size=(1,1),\
                padding='same', use_bias=False, kernel_initializer=tf.constant_initializer(layers[i+4+group].get_weights()[0]),\
                kernel_regularizer=regularizers.l2(decay), trainable=trainable)(input[:,:,:,group_index:group_index+in_group_ch]))
        else:
            tmp_list.append(ternary_conv(input[:,:,:,group_index:group_index+in_group_ch],\
                            in_group_ch, out_group_ch, k=1, wi=i+4+group, train_weights = not trainable))

    return tf.keras.layers.Concatenate()(tmp_list)



def ShuffleUnit(inp, in_ch, out_ch, groups=4, expansion=4, block=0,\
                stride=1, pix=32, decay=WDECAY, residual=True, i=0):
    global vanilla_layers, pred_fmaps_qtz_bits
    layers = vanilla_layers
    midd_ch = int(in_ch * expansion)
    #************* PW1 Group Conv **************************
    #actual pw1
    pw1 = group_conv(inp, in_ch=in_ch, out_ch=midd_ch, groups=groups, decay=decay, i=i, trainable=not train_ternary)
    #4-bit quantize for ifmap
    if pred_fmaps_qtz:
        pw1_quantized = tf.quantization.fake_quant_with_min_max_args(inp, min=-6, max=6, num_bits=4)
    else:
        pw1_quantized = inp
    pw1_tr = group_conv(pw1_quantized, in_ch=in_ch, out_ch=midd_ch, groups=groups, i=i, quantized=True, trainable=train_ternary)
    i+=9
    pw1_bn = tf.keras.layers.BatchNormalization(beta_initializer=tf.constant_initializer(layers[i].get_weights()[1]),\
                                                   gamma_initializer=tf.constant_initializer(layers[i].get_weights()[0]),\
                                                   moving_mean_initializer=tf.constant_initializer(layers[i].get_weights()[2]),\
                                moving_variance_initializer=tf.constant_initializer(layers[i].get_weights()[3]), trainable = not train_ternary)(pw1)
    pw1_tr_bn = tf.keras.layers.BatchNormalization(beta_initializer=tf.constant_initializer(layers[i].get_weights()[1]),\
                                                  gamma_initializer=tf.constant_initializer(layers[i].get_weights()[0]),\
                                                  moving_mean_initializer=tf.constant_initializer(layers[i].get_weights()[2]),\
                                 moving_variance_initializer=tf.constant_initializer(layers[i].get_weights()[3]), trainable = train_ternary)(pw1_tr)

    pw1_tr_relu = tf.keras.layers.ReLU()(pw1_tr_bn)
    approx_grad = tf.cond(train_ternary, lambda: tf.identity(pw1_tr_relu), lambda: tf.keras.layers.ReLU()(pw1_bn))
    pw1_rl = approx_grad + tf.stop_gradient(tf.where(tf.math.greater(pw1_tr_relu, 0.), pw1_bn, tf.zeros_like(pw1_bn)) - approx_grad)
    i += 2

    #*************     DW Conv    **************************
    dw = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), padding='same', strides = stride, use_bias=False,\
         depthwise_initializer=tf.constant_initializer(layers[i].get_weights()[0]), trainable = not train_ternary)(pw1_rl)
    if pred_fmaps_qtz:
        dw_quantized = tf.quantization.fake_quant_with_min_max_args(pw1_tr_relu, min=-6, max=6, num_bits=4)
    else:
        dw_quantized = pw1_tr_relu
    dw_tr = ternary_dw_conv(dw_quantized, in_ch, midd_ch, stride=stride, wi=i, train_weights=train_ternary)
    i+=1
    dw_bn = tf.keras.layers.BatchNormalization(beta_initializer=tf.constant_initializer(layers[i].get_weights()[1]),\
                                                   gamma_initializer=tf.constant_initializer(layers[i].get_weights()[0]),\
                                                   moving_mean_initializer=tf.constant_initializer(layers[i].get_weights()[2]),\
                                moving_variance_initializer=tf.constant_initializer(layers[i].get_weights()[3]), trainable = not train_ternary)(dw)
    dw_bn_tr = tf.keras.layers.BatchNormalization(beta_initializer=tf.constant_initializer(layers[i].get_weights()[1]),\
                                                   gamma_initializer=tf.constant_initializer(layers[i].get_weights()[0]),\
                                                   moving_mean_initializer=tf.constant_initializer(layers[i].get_weights()[2]),\
                                moving_variance_initializer=tf.constant_initializer(layers[i].get_weights()[3]), trainable = train_ternary)(dw_tr)
    dw_tr_relu = tf.keras.layers.ReLU()(dw_bn_tr)
    approx_grad2 = tf.cond(train_ternary, lambda: tf.identity(dw_tr_relu), lambda: tf.keras.layers.ReLU()(dw_bn))
    dw_rl = approx_grad2 + tf.stop_gradient(tf.where(tf.math.greater(dw_tr_relu, 0.), dw_bn, tf.zeros_like(dw_bn)) - approx_grad2)
    i+=2

    if residual:
        pw2 = group_conv(dw_rl, in_ch=midd_ch, out_ch=out_ch if stride==1 and in_ch==out_ch else out_ch-in_ch, groups=groups, decay=decay, i=i, trainable=not train_ternary)
    else:
        pw2 = group_conv(dw_rl, in_ch=midd_ch, out_ch=out_ch, groups=groups, decay=decay, i=i, trainable=not train_ternary)
    i+=9

    pw2_bn = tf.keras.layers.BatchNormalization(beta_initializer=tf.constant_initializer(layers[i].get_weights()[1]), \
                                                gamma_initializer=tf.constant_initializer(layers[i].get_weights()[0]),\
                                                moving_mean_initializer=tf.constant_initializer(layers[i].get_weights()[2]),\
                                                moving_variance_initializer=tf.constant_initializer(layers[i].get_weights()[3]),\
                                                trainable = train_ternary)(pw2)
    pw2_shuffled = ch_shuffle(pw2_bn, pix, out_ch, groups)
    i+=4

    if stride == 1 and in_ch == out_ch and residual:
        result = tf.math.add(pw2_shuffled, inp)
        i+=1
    elif residual:
        res = tf.keras.layers.AveragePooling2D(pool_size=3, strides=stride, padding='same')(input)
        result = tf.keras.layers.Concatenate()([pw2_shuffled, res])
        i+=2
    else:
        result = pw2_shuffled

    return result, i



def build_model():
    global WDECAY, vanilla_layers
    layers = vanilla_layers
    i = 1
    inp = tf.keras.Input(shape=(32,32,3))
    Conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', use_bias=False,\
                            kernel_initializer=tf.constant_initializer(layers[i].get_weights()[0]), kernel_regularizer=regularizers.l2(4e-5))(inp)
    i += 1
    Conv1_bn = tf.keras.layers.BatchNormalization(beta_initializer=tf.constant_initializer(layers[i].get_weights()[1]),\
                                                  gamma_initializer=tf.constant_initializer(layers[i].get_weights()[0]),\
                                                  moving_mean_initializer=tf.constant_initializer(layers[i].get_weights()[2]),\
                                                  moving_variance_initializer=tf.constant_initializer(layers[i].get_weights()[3]))(Conv1)
    Conv1_rl = tf.keras.layers.ReLU()(Conv1_bn)
    i += 2
    pw_init = group_conv(Conv1_rl, in_ch=32, out_ch=32, groups=4, decay=4e-5, i=i, trainable=True)
    i += 9
    pw_init_bn = tf.keras.layers.BatchNormalization(beta_initializer=tf.constant_initializer(layers[i].get_weights()[1]),\
                                                   gamma_initializer=tf.constant_initializer(layers[i].get_weights()[0]),\
                                                   moving_mean_initializer=tf.constant_initializer(layers[i].get_weights()[2]),\
                                                   moving_variance_initializer=tf.constant_initializer(layers[i].get_weights()[3]))(pw_init)
    pw_init_rl = tf.keras.layers.ReLU()(pw_init_bn)
    i += 2
    pw_init_shuffled = ch_shuffle(pw_init_rl, 32, 32, 4)
    i += 3
    dw_init = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), padding='same', strides=1, use_bias=False,\
                    depthwise_initializer=tf.constant_initializer(layers[i].get_weights()[0]))(pw_init_shuffled)
    i += 1
    dw_init_bn = tf.keras.layers.BatchNormalization(beta_initializer=tf.constant_initializer(layers[i].get_weights()[1]),\
                                                    gamma_initializer=tf.constant_initializer(layers[i].get_weights()[0]),\
                                                    moving_mean_initializer=tf.constant_initializer(layers[i].get_weights()[2]),\
                                                    moving_variance_initializer=tf.constant_initializer(layers[i].get_weights()[3]))(dw_init)

    i += 1
    WDECAY = 2e-4

    block1,i = ShuffleUnit(dw_init_bn, in_ch=32, out_ch=32, pix=32, expansion=74, residual=False, i=i)
    print("out")
    block2,i = ShuffleUnit(block1, in_ch=32, out_ch=32, pix=32, expansion=74, i=i)
    block3,i = ShuffleUnit(block2, in_ch=32, out_ch=32, pix=32, expansion=74, i=i)

    block4,i = ShuffleUnit(block3, in_ch=32, out_ch=64, pix=16, expansion=74, stride=2, residual=False, i=i)
    block5,i = ShuffleUnit(block4, in_ch=64, out_ch=64, pix=16, expansion=37, i=i)
    block6,i = ShuffleUnit(block5, in_ch=64, out_ch=64, pix=16, expansion=37, i=i)
    block7,i = ShuffleUnit(block6, in_ch=64, out_ch=64, pix=16, expansion=37, i=i)

    WDECAY = 1e-4
    block8,i = ShuffleUnit(block7, in_ch=64, out_ch=96, pix=8, expansion=37, stride=2, residual=False, i=i)
    block9,i = ShuffleUnit(block8, in_ch=96, out_ch=96, pix=8, expansion=36, i=i)
    block10,i = ShuffleUnit(block9, in_ch=96, out_ch=96, pix=8, expansion=36, i=i)

    block11,i = ShuffleUnit(block10, in_ch=96, out_ch=96, pix=8, expansion=36, i=i)
    block12,i = ShuffleUnit(block11, in_ch=96, out_ch=96, pix=8, expansion=36, i=i)
    block13,i = ShuffleUnit(block12, in_ch=96, out_ch=96, pix=8, expansion=36, i=i)

    block14,i = ShuffleUnit(block13, in_ch=96, out_ch=96, pix=8, expansion=36, i=i)
    block15,i = ShuffleUnit(block14, in_ch=96, out_ch=96, pix=8, expansion=36, i=i)
    block16,i = ShuffleUnit(block15, in_ch=96, out_ch=160, pix=8, expansion=36, residual=False, i=i)

    pw_final = group_conv(block16, in_ch=160, out_ch=960, groups=4, trainable=True, i=i)
    i+=9
    pw_final_bn = tf.keras.layers.BatchNormalization(beta_initializer=tf.constant_initializer(layers[i].get_weights()[1]), gamma_initializer=tf.constant_initializer(layers[i].get_weights()[0]),\
                         moving_mean_initializer=tf.constant_initializer(layers[i].get_weights()[2]), moving_variance_initializer=tf.constant_initializer(layers[i].get_weights()[3]))(pw_final)
    pw_final_rl = tf.keras.layers.ReLU()(pw_final_bn)
    global_ave_pool = tf.keras.layers.GlobalAveragePooling2D()(pw_final_rl)
    i += 3
    dense = tf.keras.layers.Dense(10, activation='softmax', use_bias=True, kernel_initializer =tf.constant_initializer(layers[i].get_weights()[0]),\
                  bias_initializer = tf.constant_initializer(layers[i].get_weights()[1]), kernel_regularizer=regularizers.l2(4e-5))(global_ave_pool)
    return tf.keras.models.Model(inputs=inp, outputs=dense)

#####################################################################################################
#Learing rate & checkpoints
def generate_learning_rates():
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
    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_name,
                monitor='val_accuracy', verbose=1, save_best_only=True,\
                save_weights_only=False, mode='auto', save_freq='epoch')
    checkpoint2 = tf.keras.callbacks.ModelCheckpoint("pred2_lastepoch.h5",
                monitor='val_accuracy', verbose=1, save_best_only=False,\
                save_weights_only=False, mode='auto', save_freq='epoch')
    return checkpoint, checkpoint2

def get_callbacks():
    learning_rates = generate_learning_rates()
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
    global vanilla_layers,  DATA_DIR, training, pre_trained, TOTAL_GPUS, ACTIVE_GPUS
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
            optimizer = tf.keras.optimizers.SGD(learning_rate=INIT_LEARNING_RATE, momentum=0.9)
            model.compile(loss=LOSS, optimizer=optimizer, metrics=ACCURACY)
        else:
            vanilla_layers = tf.keras.models.load_model(pruned_model).layers
            model = build_model()
            model.load_weights(pred_model_lastepoch, by_name=True)
            optimizer = tf.keras.optimizers.SGD(learning_rate=INIT_LEARNING_RATE, momentum=0.9)
            model.compile(loss=LOSS, optimizer=optimizer, metrics=ACCURACY)
            #model = tf.keras.models.load_model(pred_model)
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
