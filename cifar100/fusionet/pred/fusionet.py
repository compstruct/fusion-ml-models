
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
ref_model = "pruned_named.h5"
model_name = "pred.h5"

#was trained using 4 GPUs
TOTAL_GPUS = 4
ACTIVE_GPUS = 4
GPU_MEM = 9.8e3
CPUs = 8

IMG_SIZE = IMG_H = IMG_W = 32
BATCH_SIZE = 16 * ACTIVE_GPUS
EPOCHS = 50
CLASSES = 100
VERBOSE = 1
WDECAY = 1e-4
INIT_LEARNING_RATE = 2e-2

training = False
pre_trained = True
model_summary = False
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







@tf.function
def compute_threshold(x, my_name=" "):
    x_sum = tf.reduce_sum(tf.abs(x, name=my_name+"comth_ab1"), keepdims =False, name=my_name+"comth_redsum")
    size = tf.cast(tf.size(x, name=my_name+"comth_siz"), tf.float32, name=my_name+"comth_cas")
    threshold = tf.math.scalar_mul(0.7/size, x_sum, name=my_name+"comth_mul")
    return threshold

@tf.function
def compute_alpha(x, my_name=" "):
    threshold = compute_threshold(x, my_name=my_name+"thr1")
    #Weights above delta
    alpha1_temp1 = tf.stop_gradient(tf.where(tf.greater(x, threshold, name=my_name+"comalph_great1"), x, tf.zeros_like(x, tf.float32, name=my_name+"comalph_zero1"), name=my_name+"comalph_wher1"))
    #Weights below negative delta
    alpha1_temp2 = tf.stop_gradient(tf.where(tf.less(x,-threshold, name=my_name+"comalph_less1"), x, tf.zeros_like(x, tf.float32, name=my_name+"comalph_zero2"), name=my_name+"comalph_wher2"))
    #replace all the non-zero elements with 1
    alpha_array = tf.math.add(alpha1_temp1, alpha1_temp2, name=my_name+"comalph_add1")
    alpha_array_abs = tf.math.abs(alpha_array, name=my_name+"comalph_abs1")
    alpha_array_abs1 = tf.stop_gradient(tf.where(tf.greater(alpha_array_abs,0, name=my_name+"comalph_greater2"),tf.ones_like(alpha_array_abs,tf.float32, name=my_name+"comalph_ones1"), tf.zeros_like(alpha_array_abs, tf.float32, name=my_name+"comalph_zero3"), name=my_name+"comalph_abs2"))

    #divide actual sum of elements with size of non-zeros
    alpha_sum = tf.reduce_sum(alpha_array_abs, name=my_name+"comalph_redsum1")
    n = tf.reduce_sum(alpha_array_abs1, name=my_name+"comalph_sum2")
    alpha = tf.math.divide(alpha_sum,n, name=my_name+"comalph_div1")
    return alpha

@tf.function
def ternerize(x, my_name=" "):
    g = tf.compat.v1.get_default_graph()
    #with ops.name_scope("tenarized") as name:
    with g.gradient_override_map({"Sign": "Identity"}):
        threshold =compute_threshold(x, my_name=my_name+"thr")
        x=tf.sign(tf.add(tf.sign(tf.add(x,threshold, name=my_name+"tern_add1")),tf.sign(tf.add(x,-threshold, name=my_name+"tern_sig1"),\
                  name=my_name+"tern_sig2"), name=my_name+"tern_add2"), name=my_name+"tern_sig3")
        return x



class Ternarize(tf.keras.constraints.Constraint):

  def __init__(self, name=""):
    super(Ternarize, self).__init__()
    self.name=name

  def __call__(self, w):
    my_name = "constraint"+self.name
    alpha_w = compute_alpha(w, my_name=my_name)
    ter_w = ternerize(w, my_name=my_name)
    ter_w_alpha = tf.multiply(alpha_w, ter_w)
    return ter_w_alpha

  def get_config(self):
    config = super(Ternarize, self).get_config()
    config.update({"name": self.name})
    return config




def ch_shuffle(x, pix, in_ch, groups, name=""):
    ch_per_group = in_ch // groups
    x_reshaped = tf.keras.layers.Reshape((pix, pix, groups, ch_per_group), name=name+"resh1")(x)
    x_transposed = tf.keras.layers.Permute((1,2,4,3), name=name+"perm1")(x_reshaped)
    x_shuffled = tf.keras.layers.Reshape((pix, pix, in_ch), name=name+"resh2")(x_transposed)
    return x_shuffled



def group_conv(input, in_ch, out_ch, groups, decay=WDECAY, name="", trainable=True):
    in_group_ch = in_ch // groups
    out_group_ch = out_ch // groups
    group_list = []

    for group in range(groups):
        group_index = group * in_group_ch
        group_conv = tf.keras.layers.Conv2D(filters=out_group_ch, kernel_size=(1,1),\
           padding='same', use_bias=False, kernel_initializer="he_normal",\
           kernel_regularizer=regularizers.l2(decay), trainable=trainable, name=name+"g"+str(group))(input[:,:,:,group_index:group_index+in_group_ch])
        group_list.append(group_conv)
    g_conv_res = tf.keras.layers.Concatenate(name=name+"groupconc")(group_list)
    return g_conv_res


def group_conv_tr(input, in_ch, out_ch, groups, decay=WDECAY, name="", trainable=True):
    in_group_ch = in_ch // groups
    out_group_ch = out_ch // groups
    group_list = []

    for group in range(groups):
        group_index = group * in_group_ch
        group_conv = tf.keras.layers.Conv2D(filters=out_group_ch, kernel_size=(1,1),\
           padding='same', use_bias=False, kernel_initializer="he_normal",\
           kernel_regularizer=regularizers.l2(decay), trainable=trainable,\
           kernel_constraint=Ternarize(name=name+"g"+str(group)), name=name+"g"+str(group))(input[:,:,:,group_index:group_index+in_group_ch])

        group_list.append(group_conv)

    g_conv_res = tf.keras.layers.Concatenate(name=name+"groupconc")(group_list)
    return g_conv_res



def ShuffleUnit(input, in_ch, out_ch, groups=4, expansion=4, stride=1, pix=32, decay=WDECAY, residual=True, name=" "):

    midd_ch = int(in_ch * expansion)
    #************* PW1 Group Conv **************************
    pw1 = group_conv(input, in_ch=in_ch, out_ch=midd_ch, groups=groups, decay=decay, name=name+"conv1", trainable=False)
    pw1_bn = tf.keras.layers.BatchNormalization(name=name+"bn1", trainable=False)(pw1)

    if pred_fmaps_qtz:
        pw1_quantized = tf.quantization.fake_quant_with_min_max_args(input, min=-6, max=6, num_bits=4, name=name+"quantz1")
    else:
        pw1_quantized = input

    pw1_tr = group_conv_tr(pw1_quantized, in_ch=in_ch, out_ch=midd_ch, groups=groups, decay=decay, name=name+"trconv1", trainable=True)
    pw1_bn_tr = tf.keras.layers.BatchNormalization(name=name+"trbn1", trainable=True)(pw1_tr)
    pw1_tr_rl = tf.keras.layers.ReLU(name=name+"trrl1")(pw1_bn_tr)
    pw1_approx_grad = tf.identity(pw1_tr_rl, name=name+"triden1")
    pw1_rl = pw1_approx_grad + tf.stop_gradient(tf.where(tf.math.greater(pw1_tr_rl, 0., name=name+"tr_rl_great1"),\
                                        pw1_bn, tf.zeros_like(pw1_bn, name=name+"tr_tl_zero1"), name=name+"tr_rl_where1")\
                                        - pw1_approx_grad, name=name+"tr_rl_stopgrad1")

    #*************     DW Conv    **************************
    dw = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), padding='same', strides = stride,
                                         use_bias=False, kernel_initializer="he_normal", name=name+"conv2", trainable=False)(pw1_rl)
    dw_bn = tf.keras.layers.BatchNormalization(name=name+"bn2", trainable=False)(dw)

    if pred_fmaps_qtz:
        dw_quantized = tf.quantization.fake_quant_with_min_max_args(pw1_tr_rl, min=-6, max=6, num_bits=4, name=name+"quantz2")
    else:
        dw_quantized = pw1_tr_rl
    dw_tr = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), padding='same', strides = stride,
                                         use_bias=False, kernel_initializer="he_normal", name=name+"trconv2", depthwise_constraint=Ternarize(name=name+"trconv2"), trainable=True)(dw_quantized)

    dw_bn_tr = tf.keras.layers.BatchNormalization(name=name+"trbn2", trainable=True)(dw_tr)
    dw_tr_rl = tf.keras.layers.ReLU(name=name+"trrl2")(dw_bn_tr)
    dw_approx_grad = tf.identity(dw_tr_rl, name=name+"triden2")
    dw_rl = dw_approx_grad + tf.stop_gradient(tf.where(tf.math.greater(dw_tr_rl, 0., name=name+"tr_rl_great2"),\
                                        dw_bn, tf.zeros_like(dw_bn, name=name+"tr_tl_zero2"), name=name+"tr_rl_where2")\
                                        - dw_approx_grad, name=name+"tr_rl_stopgrad2")


    #dw_rl = tf.keras.layers.ReLU(name=name+"rl2")(dw_bn)
    if residual:
        pw2 = group_conv(dw_rl, in_ch=midd_ch, out_ch=out_ch if stride==1 and in_ch==out_ch else out_ch-in_ch, groups=groups, decay=decay, name=name+"conv3", trainable=False)
    else:
        pw2 = group_conv(dw_rl, in_ch=midd_ch, out_ch=out_ch, groups=groups, decay=decay, name=name+"conv3", trainable=False)
    pw2_bn = tf.keras.layers.BatchNormalization(name=name+"bn3")(pw2)
    pw2_shuffled = ch_shuffle(pw2_bn, pix, out_ch, groups, name=name+"shuff1")

    if stride == 1 and in_ch == out_ch and residual:
        result = tf.math.add(pw2_shuffled, input, name=name+"add1")
    elif residual:
        res = tf.keras.layers.AveragePooling2D(pool_size=3, strides=stride, padding='same', name=name+"avepool1")(input)
        result = tf.keras.layers.Concatenate(name=name+"conca1")([pw2_shuffled, res])
    else:
        result = pw2_shuffled

    return result



def build_model():
    global WDECAY
    inp = tf.keras.Input(shape=(32,32,3))

    Conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', use_bias=False,
                                   kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5), name="conv1")(inp)

    Conv1_bn = tf.keras.layers.BatchNormalization(name="bn1")(Conv1)
    Conv1_rl = tf.keras.layers.ReLU(name="rl1")(Conv1_bn)

    pw_init = group_conv(Conv1_rl, in_ch=32, out_ch=32, groups=4, decay=4e-5, name="conv2")
    pw_init_bn = tf.keras.layers.BatchNormalization(name="bn2")(pw_init)
    pw_init_rl = tf.keras.layers.ReLU(name="rl2")(pw_init_bn)
    pw_init_shuffled = ch_shuffle(pw_init_rl, 32, 32, 4, name="shuff1")
    dw_init = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), padding='same', strides = 1, use_bias=False,
                                              kernel_initializer="he_normal", name="conv3")(pw_init_shuffled)
    dw_init_bn = tf.keras.layers.BatchNormalization(name="bn3")(dw_init)

    WDECAY = 2e-4
    block1 = ShuffleUnit(input=dw_init_bn, in_ch=32, out_ch=32, pix=32, expansion=74, residual=False, name="b1")
    block2 = ShuffleUnit(input=block1, in_ch=32, out_ch=32, pix=32, expansion=74, name="b2")
    block3 = ShuffleUnit(input=block2, in_ch=32, out_ch=32, pix=32, expansion=74, name="b3")

    block4 = ShuffleUnit(input=block3, in_ch=32, out_ch=64, pix=16, expansion=74, stride=2, residual=False, name="b4")
    block5 = ShuffleUnit(input=block4, in_ch=64, out_ch=64, pix=16, expansion=37, name="b5")
    block6 = ShuffleUnit(input=block5, in_ch=64, out_ch=64, pix=16, expansion=37, name="b6")
    block7 = ShuffleUnit(input=block6, in_ch=64, out_ch=64, pix=16, expansion=37, name="b7")

    WDECAY = 1e-4
    block8 = ShuffleUnit(input=block7, in_ch=64, out_ch=96, pix=8, expansion=37, stride=2, residual=False, name="b8")
    block9 = ShuffleUnit(input=block8, in_ch=96, out_ch=96, pix=8, expansion=36, name="b9")
    block10 = ShuffleUnit(input=block9, in_ch=96, out_ch=96, pix=8, expansion=36, name="b10")

    block11 = ShuffleUnit(input=block10, in_ch=96, out_ch=96, pix=8, expansion=36, name="b11")
    block12 = ShuffleUnit(input=block11, in_ch=96, out_ch=96, pix=8, expansion=36, name="b12")
    block13 = ShuffleUnit(input=block12, in_ch=96, out_ch=96, pix=8, expansion=36, name="b13")

    block14 = ShuffleUnit(input=block13, in_ch=96, out_ch=96, pix=8, expansion=36, name="b14")
    block15 = ShuffleUnit(input=block14, in_ch=96, out_ch=96, pix=8, expansion=36, name="b15")
    block16 = ShuffleUnit(input=block15, in_ch=96, out_ch=160, pix=8, expansion=36, residual=False, name="b16")

    expanded_lastConv = tf.keras.layers.Conv2D(filters=1280, kernel_size=(1,1), padding='same',
                 use_bias=False, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(WDECAY), name="conv4")(block16)
    #pw_final = group_conv(block16, in_ch=160, out_ch=1280, groups=4, name="conv4")
    pw_final_bn = tf.keras.layers.BatchNormalization(name="bn4")(expanded_lastConv)
    pw_final_rl = tf.keras.layers.ReLU(name="rl3")(pw_final_bn)

    global_ave_pool = tf.keras.layers.GlobalAveragePooling2D(name="ave_pool")(pw_final_rl)
    dense = tf.keras.layers.Dense(100, activation='softmax', use_bias=True, kernel_initializer = 'glorot_uniform',
                                  bias_initializer = 'zeros', kernel_regularizer=regularizers.l2(4e-5), name="dense1")(global_ave_pool)
    return tf.keras.models.Model(inputs=inp, outputs=dense)









#####################################################################################################
#Learing rate & checkpoints
def generate_learning_rates1():
    learning_rates=[]
    lr = 0.0002
    decay_steps = 42
    alpha = 0.001
    total_epochs = EPOCHS
    initial_learning_rate = 0.01
    end_learning_rate=0.0001

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
            ref = tf.keras.models.load_model(ref_model)

            common_names = ["conv1", "bn1", "conv2g0",  "conv2g1",  "conv2g2",  "conv2g3", "bn2", "conv3", "bn3", "conv4", "bn4", "dense1"]
            for blk in range(1,17):
                blk_name = "b"+str(blk)
                for group in range(4):
                    common_names.append(blk_name+"conv1g"+str(group))
                common_names.append(blk_name+"bn1")
                common_names.append(blk_name+"conv2")
                common_names.append(blk_name+"bn2")
                for group in range(4):
                    common_names.append(blk_name+"conv3g"+str(group))
                common_names.append(blk_name+"bn3")

            for common_name in common_names:
                model.get_layer(common_name).set_weights(ref.get_layer(common_name).get_weights())

            for blk in range(1,17):
                blk_name = "b"+str(blk)
                model.get_layer(blk_name+"trbn1").set_weights(ref.get_layer(blk_name+"bn1").get_weights())
                model.get_layer(blk_name+"trbn2").set_weights(ref.get_layer(blk_name+"bn2").get_weights())
                model.get_layer(blk_name+"trconv2").set_weights(ref.get_layer(blk_name+"conv2").get_weights())

            for blk in range(1,17):
                blk_name = "b"+str(blk)
                for group in range(4):
                    initializer = tf.constant_initializer(ref.get_layer(blk_name+"conv1g"+str(group)).get_weights()[0])
                    my_w = tf.Variable(initializer(shape=ref.get_layer(blk_name+"conv1g"+str(group)).get_weights()[0].shape), name=blk_name+"wtrconv1g"+str(group))
                    alpha_w = compute_alpha(my_w, my_name=blk_name+"trconv1g"+str(group))
                    ter_alpha = alpha_w
                    ter_w = ternerize(my_w,  my_name=blk_name+"trconv1g"+str(group))
                    ter_w_alpha = tf.multiply(alpha_w, ter_w, name=blk_name+"trconv1g"+str(group)+"_alpha_mul")
                    ter_w_alpha_expanded = tf.expand_dims(ter_w_alpha, axis=0)
                    model.get_layer(blk_name+"trconv1g"+str(group)).set_weights(ter_w_alpha_expanded)

            optimizer = tf.keras.optimizers.SGD(learning_rate=INIT_LEARNING_RATE, momentum=0.9)
            #optimizer = tf.keras.optimizers.Adam(learning_rate=INIT_LEARNING_RATE)
            model.compile(loss=LOSS, optimizer=optimizer, metrics=ACCURACY)


        else:
            model = tf.keras.models.load_model("pred_last_epoch.h5", custom_objects={'Ternarize' : Ternarize})
        if training:
            #print(model.get_layer("b1trconv1g0").get_weights())
            #print(model.get_layer("b1trconv1g1").get_weights())

            model.fit(train,
                epochs=EPOCHS,
                steps_per_epoch=int(train_size / BATCH_SIZE),
                validation_data=val,
                validation_steps=int(val_size / BATCH_SIZE),
                validation_freq=1,
                callbacks=get_callbacks())

            model.save('pred_last_epoch.h5')

            #print(model.get_layer("b1trconv1g0").get_weights())
            #print(model.get_layer("b1trconv1g1").get_weights())
        else:
            #print(model.get_layer("b1trconv1g0").get_weights())
            #print(model.get_layer("b3trconv1g1").get_weights())

            #print(model.get_layer("b1trconv2").get_weights())
            #print(model.get_layer("b2trconv2").get_weights())
            #print(model.get_layer("b3trconv2").get_weights())
            #print(model.get_layer("b4trconv2").get_weights())

            model.evaluate(val, steps=int(val_size/BATCH_SIZE))
            #model.summary()
#####################################################################################################



main()
