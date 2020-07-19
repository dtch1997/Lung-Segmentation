# Compatibility with TF 2.0--this is TF 1 code
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except:
    import tensorflow as tf

from .layers import *

import scipy.ndimage
import logging
logging.basicConfig(level=logging.INFO)

def general_conv(layer, is_training, architecture_conv, name="general_conv"):
    """
    A generalized convolution block that takes an architecture.
    INPUTS:
    - layer: (tensor.4d) input tensor.
    - is_training: (bool) are we in training size
    - architecture_conv: (list of lists)
      [[filt_size, filt_num, stride], ..., [0, poolSize],
       [filt_size, filt_num, stride], ..., [0, poolSize],
       ...]
    - b_name: (string) branch name.  If not doing branch, doesn't matter.
    """
    for conv_iter, conv_numbers in enumerate(architecture_conv):
        if conv_numbers[0]==0:
            layer = max_pool3d(layer, k=conv_numbers[1])
        else:
            if len(conv_numbers)==2:
                conv_numbers.append(1)
            layer = conv3d_bn_relu(layer, is_training, conv_numbers[0], conv_numbers[1], stride=conv_numbers[2],
                           name=(name+"_conv"+str(conv_iter)))
    return layer

def Le_Net(layer, is_training, class_num, batch_size, name="Le_Net"):
    """
    This is the famous LeNet incarnation of the inception network.
    All the power is in the convs, so this is quite simple.
    INPUTS:
    - layer: (tensor.4d) input tensor.
    - output_size: (int) number of classes we're predicting
    - keep_prob: (float) probability to keep during dropout.
    - name: (str) the name of the network
    """
    architecture_conv = [[5,6],[0,2],
                         [5,16],[0,2]]
    layer = general_conv(layer, is_training, architecture_conv, name=name)
    layer = deconv3d_w_bias(layer, 4, class_num, batch_size, name=name+"_deconv")
    return layer

def Alex_Net(layer, is_training, class_num, batch_size, name="Alex_Net"):
    """
    This is the famous AlexNet incarnation of the inception network.
    All the power is in the convs, so this is quite simple.
    INPUTS:
    - layer: (tensor.4d) input tensor.
    - output_size: (int) number of classes we're predicting
    - keep_prob: (float) probability to keep during dropout.
    - name: (str) the name of the network
    """
    architecture_conv = [[11,96,4],[0,2],
                         [11,256],[0,2],
                         [3,384],[3,384],[3,256],[0,2]]
    layer = general_conv(layer, is_training, architecture_conv, name=name)
    layer = deconv3d_w_bias(layer, 32, class_num, batch_size, name=name+"_deconv")
    return layer

def VGG11_Net(layer, is_training, class_num, batch_size, name="VGG11_Net"):
    """
    This is the 11-layer VGG Network.
    INPUTS:
    - layer: (tensor.4d) input tensor.
    - output_size: (int) number of classes we're predicting
    - keep_prob: (float) probability to keep during dropout.
    - name: (str) the name of the network
    """
    architecture_conv = [[3,64],[0,2],
                         [3,128],[0,2],
                         [3,256],[3,256],[0,2],
                         [3,512],[3,512],[0,2],
                         [3,512],[3,512],[0,2]]
    layer = general_conv(layer, is_training, architecture_conv, name=name)
    layer = deconv3d_w_bias(layer, 32, class_num, batch_size, name=name+"_deconv")
    return layer

def VGG13_Net(layer, is_training, class_num, batch_size, name="VGG13_Net"):
    """
    This is the 13-layer VGG Network.
    INPUTS:
    - layer: (tensor.4d) input tensor.
    - output_size: (int) number of classes we're predicting
    - keep_prob: (float) probability to keep during dropout.
    - name: (str) the name of the network
    """
    architecture_conv = [[3,64],[3,64],[0,2],
                         [3,128],[3,128],[0,2],
                         [3,256],[3,256],[0,2],
                         [3,512],[3,512],[0,2],
                         [3,512],[3,512],[0,2]]
    layer = general_conv(layer, is_training, architecture_conv, name=name)
    layer = deconv3d_w_bias(layer, 32, class_num, batch_size, name=name+"_deconv")
    return layer

def VGG16_Net(layer, is_training, class_num, batch_size, name="VGG16_Net"):
    """
    This is the 16-layer VGG Network.
    INPUTS:
    - layer: (tensor.4d) input tensor.
    - output_size: (int) number of classes we're predicting
    - keep_prob: (float) probability to keep during dropout.
    - name: (str) the name of the network
    """
    architecture_conv = [[3,64],[3,64],[0,2],
                         [3,128],[3,128],[0,2],
                         [3,256],[3,256],[3,256],[0,2],
                         [3,512],[3,512],[3,512],[0,2],
                         [3,512],[3,512],[3,512],[0,2]]
    layer = general_conv(layer, is_training, architecture_conv, name=name)
    layer = deconv3d_w_bias(layer, 32, class_num, batch_size, name=name+"_deconv")
    return layer

def VGG19_Net(layer, is_training, class_num, batch_size, name="VGG19_Net"):
    """
    This is the 19-layer VGG Network.
    INPUTS:
    - layer: (tensor.4d) input tensor.
    - output_size: (int) number of classes we're predicting
    - keep_prob: (float) probability to keep during dropout.
    - name: (str) the name of the network
    """
    architecture_conv = [[3,64],[3,64],[0,2],
                         [3,128],[3,128],[0,2],
                         [3,256],[3,256],[3,256],[3,256],[0,2],
                         [3,512],[3,512],[3,512],[3,512],[0,2],
                         [3,512],[3,512],[3,512],[3,512],[0,2]]
    layer = general_conv(layer, is_training, architecture_conv, name=name)
    layer = deconv3d_w_bias(layer, 32, class_num, batch_size, name=name+"_deconv")
    return layer

def inceptionv1_module_split(layer, is_training, kSize=[16,16,16,16,16,16], name="inceptionv1_module"):
    """
    So, this is the classical incept layer.
    INPUTS:
    - layer: (tensor.4d) input of size [batch_size, layer_width, layer_height, channels]
    - is_training: (bool) are we in training size
    - ksize: (array (6,)) [1x1, 3x3reduce, 3x3, 5x5reduce, 5x5, poolproj]
    - name: (string) name of incept layer
    """
    layer_1x1 = conv3d_bn_relu(layer, is_training, 1, kSize[0], name=(name+"_1x1"))
    layer_3x3a = conv3d_bn_relu(layer, is_training, 1, kSize[1], name=(name+"_3x3a"))
    layer_3x3b = conv3d_bn_relu(layer_3x3a, is_training, 3, kSize[2], name=(name+"_3x3b"))
    layer_5x5a = conv3d_bn_relu(layer, is_training, 1, kSize[3], name=(name+"_5x5a"))
    layer_5x5b = conv3d_bn_relu(layer_5x5a, is_training, 5, kSize[4], name=(name+"_5x5b"))
    layer_poola = max_pool3d(layer, k=3, stride=1)
    layer_poolb = conv3d_bn_relu(layer_poola, is_training, 1, kSize[5], name=(name+"_poolb"))
    l1_0,l1_1 = tf.split(layer_1x1, 2, axis=4)
    l2_0,l2_1 = tf.split(layer_3x3b, 2, axis=4)
    l3_0,l3_1 = tf.split(layer_5x5b, 2, axis=4)
    l4_0,l4_1 = tf.split(layer_poolb, 2, axis=4)
    
    return tf.concat([l1_0, l2_0, l3_0, l4_0], 4),tf.concat([l1_1, l2_1, l3_1, l4_1], 4)

def inceptionv1_module(layer, is_training, kSize=[16,16,16,16,16,16], with_batch_norm=False, name="inceptionv1_module"):
    """
    So, this is the classical incept layer.
    INPUTS:
    - layer: (tensor.4d) input of size [batch_size, layer_width, layer_height, channels]
    - is_training: (bool) are we in training size
    - ksize: (array (6,)) [1x1, 3x3reduce, 3x3, 5x5reduce, 5x5, poolproj]
    - name: (string) name of incept layer
    """
    layer_1x1 = conv3d_bn_relu(layer, is_training, 1, kSize[0], with_batch_norm=with_batch_norm, name=(name+"_1x1"))
    layer_3x3a = conv3d_bn_relu(layer, is_training, 1, kSize[1], with_batch_norm=with_batch_norm, name=(name+"_3x3a"))
    layer_3x3b = conv3d_bn_relu(layer_3x3a, is_training, 3, kSize[2], with_batch_norm=with_batch_norm, name=(name+"_3x3b"))
    layer_5x5a = conv3d_bn_relu(layer, is_training, 1, kSize[3], with_batch_norm=with_batch_norm, name=(name+"_5x5a"))
    layer_5x5b = conv3d_bn_relu(layer_5x5a, is_training, 5, kSize[4], with_batch_norm=with_batch_norm, name=(name+"_5x5b"))
    layer_poola = max_pool3d(layer, k=3, stride=1)
    layer_poolb = conv3d_bn_relu(layer_poola, is_training, 1, kSize[5], with_batch_norm=with_batch_norm, name=(name+"_poolb"))
    logging.info("Incept layer: outputs %d" % (kSize[0] + kSize[2] + kSize[4] + kSize[5]))
    return tf.concat([layer_1x1, layer_3x3b, layer_5x5b, layer_poolb], 4)

def DoG_filters(layer):
    layer_rt2 = np.zeros_like(layer)
    layer_2 = np.zeros_like(layer)
    layer_2rt2 = np.zeros_like(layer)
    layer_4 = np.zeros_like(layer)
    layer_4rt2 = np.zeros_like(layer)
    layer_8 = np.zeros_like(layer)
    layer_8rt2 = np.zeros_like(layer)
    layer_16 = np.zeros_like(layer)
    layer_16rt2 = np.zeros_like(layer)
    for i in range(layer.shape[0]):
        layer_rt2[i,:,:,:,0] = scipy.ndimage.filters.gaussian_filter(layer[i,:,:,:,0], sigma=np.sqrt(2))
        layer_2[i,:,:,:,0] = scipy.ndimage.filters.gaussian_filter(layer_rt2[i,:,:,:,0], sigma=np.sqrt(2))
        layer_2rt2[i,:,:,:,0] = scipy.ndimage.filters.gaussian_filter(layer_2[i,:,:,:,0], sigma=np.sqrt(2))
        layer_4[i,:,:,:,0] = scipy.ndimage.filters.gaussian_filter(layer_2rt2[i,:,:,:,0], sigma=np.sqrt(2))
        layer_4rt2[i,:,:,:,0] = scipy.ndimage.filters.gaussian_filter(layer_4[i,:,:,:,0], sigma=np.sqrt(2))
        layer_8[i,:,:,:,0] = scipy.ndimage.filters.gaussian_filter(layer_4rt2[i,:,:,:,0], sigma=np.sqrt(2))
        layer_8rt2[i,:,:,:,0] = scipy.ndimage.filters.gaussian_filter(layer_8[i,:,:,:,0], sigma=np.sqrt(2))
        layer_16[i,:,:,:,0] = scipy.ndimage.filters.gaussian_filter(layer_8rt2[i,:,:,:,0], sigma=np.sqrt(2))
        layer_16rt2[i,:,:,:,0] = scipy.ndimage.filters.gaussian_filter(layer_16[i,:,:,:,0], sigma=np.sqrt(2))
    layer = np.concatenate((layer,
                            layer_2-layer_rt2,
                            layer_2rt2-layer_2,
                            layer_4-layer_2rt2,
                            layer_4rt2-layer_4,
                            layer_8-layer_4rt2,
                            layer_8rt2-layer_8,
                            layer_16-layer_8rt2,
                            layer_16rt2-layer_16), axis=4)
    return layer


def dense_conv(layer, is_training, k=32, name="dense_conv"):
    layer_conv = layer
    layer_bn = batch_norm(layer_conv, is_training, name=name+"_bn1x1")
    layer_relu = tf.maximum(layer_bn, 0.0)
    layer_conv = conv3d_wo_bias(layer_relu, 1, k*4, name=name+"_conv1x1")
    layer_bn = batch_norm(layer_conv, is_training, name=name+"_bn3x3")
    layer_relu = tf.maximum(layer_bn, 0.0)
    layer_conv = conv3d_wo_bias(layer_relu, 3, k, name=name+"_conv3x3")
    return tf.concat([layer,layer_conv], 4)

def upsample_concat(layer, block, t, is_training, batch_size, name="upsample_concat"):
    layer = batch_norm(layer, is_training, name=name+"_bn")
    layer = tf.maximum(layer, 0.0)
    size_layer = layer.get_shape().as_list()
    #layer = deconv3d_w_bias(layer, 2, int(size_layer[4]*t), batch_size, name=name+"_deconv")
    layer = deconv3d_w_bias(layer, 2, 32, batch_size, name=name+"_deconv")
    return tf.concat([block, layer], 4)

def simple_denseU_Net(layer, is_training, keep_prob, class_num, batch_size, name="simple_denseU_Net"):
    seg1 = 0
    seg2 = 0
    seg3 = 0
    seg4 = 0
    k = 32
    layer = conv3d_wo_bias(layer, 7, k, name=name+"_conv1")
    for i in range(15):
        layer = dense_conv(layer, is_training, k=k, name=name+"_dense_"+str(i))
    seg1 = batch_norm(layer, is_training, name=name+"_seg1_bn")
    seg1 = tf.maximum(seg1, 0.0)
    seg1 = conv3d_w_bias(layer, 1, class_num, name=name+"_seg1_out")
    return seg1,seg1,seg1,seg1
    

def denseU_Net(layer, is_training, keep_prob, class_num, batch_size, name="denseU_Net"):
    seg1 = 0
    seg2 = 0
    seg3 = 0
    seg4 = 0
    k=32
    t=0.5
    layer = conv3d_wo_bias(layer, 7, k, name=name+"_conv1")
    # First Block
    for i in range(3):
        layer = dense_conv(layer, is_training, k=k, name=name+"_dense1_"+str(i))
    block1 = layer
    size_layer = layer.get_shape().as_list()
    layer = conv3d_wo_bias(layer, 1, size_layer[4]*t, name=name+"_compress1")
    layer = tf.nn.avg_pool3d(layer, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='VALID')
    # Second Block
    for i in range(6):
        layer = dense_conv(layer, is_training, k=k, name=name+"_dense2_"+str(i))
    block2 = layer
    size_layer = layer.get_shape().as_list()
    layer = conv3d_wo_bias(layer, 1, size_layer[4]*t, name=name+"_compress2")
    layer = tf.nn.avg_pool3d(layer, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='VALID')
    # Third Block
    for i in range(12):
        layer = dense_conv(layer, is_training, k=k, name=name+"_dense3_"+str(i))
    block3 = layer
    size_layer = layer.get_shape().as_list()
    layer = conv3d_wo_bias(layer, 1, size_layer[4]*t, name=name+"_compress3")
    layer = tf.nn.avg_pool3d(layer, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='VALID')
    # Fourth Block
    for i in range(16):
        layer = dense_conv(layer, is_training, k=k, name=name+"_dense4_"+str(i))
    seg4 = batch_norm(layer, is_training, name=name+"_seg4_bn")
    seg4 = tf.maximum(seg4, 0.0)
    seg4 = deconv3d_w_bias(seg4, 8, class_num, batch_size, name=name+"_seg4_out")

    # Fifth Block
    layer = upsample_concat(layer, block3, t, is_training, batch_size, name=name+"_upsample5")
    for i in range(12):
        layer = dense_conv(layer, is_training, k=k, name=name+"_dense5_"+str(i))
    seg3 = batch_norm(layer, is_training, name=name+"_seg3_bn")
    seg3 = tf.maximum(seg3, 0.0)
    seg3 = deconv3d_w_bias(seg3, 4, class_num, batch_size, name=name+"_seg3_out")
    # Sixth Block
    layer = upsample_concat(layer, block2, t, is_training, batch_size, name=name+"_upsample6")
    for i in range(6):
        layer = dense_conv(layer, is_training, k=k, name=name+"_dense6_"+str(i))
    seg2 = batch_norm(layer, is_training, name=name+"_seg2_bn")
    seg2 = tf.maximum(seg2, 0.0)
    seg2 = deconv3d_w_bias(seg2, 2, class_num, batch_size, name=name+"_seg2_out")
    # Seventh Block
    layer = upsample_concat(layer, block1, t, is_training, batch_size, name=name+"_upsample7")
    for i in range(3):
        layer = dense_conv(layer, is_training, k=k, name=name+"_dense7_"+str(i))
    seg1 = batch_norm(layer, is_training, name=name+"_seg1_bn")
    seg1 = tf.maximum(seg1, 0.0)
    seg1 = deconv3d_w_bias(seg1, 1, class_num, batch_size, name=name+"_seg1_out")
    return seg1,seg2,seg3,seg4


def chestCTU_Net(layer, is_training, keep_prob, class_num, batch_size, name="chestCTU_Net"):
    seg1 = 0
    seg2 = 0
    seg3 = 0
    seg4 = 0
    # First Block (-)
    b1l1 = conv3d_bn_relu(layer, is_training, 7, 64, name=name+"_block1conv1")
    b1l2 = conv3d_bn_relu(b1l1, is_training, 3, 64, name=name+"_block1conv2")
    # Second Block (V)
    b2l1 = conv3d_bn_relu(b1l2, is_training, 3, 128, stride=2, name=name+"_block2conv1")
    b2l2 = conv3d_bn_relu(b2l1, is_training, 3, 128, name=name+"_block2conv2")
    # Third Block (V)
    b3l1 = conv3d_bn_relu(b2l2, is_training, 3, 256, stride=2, name=name+"_block3conv1")
    b3l2 = conv3d_bn_relu(b3l1, is_training, 3, 256, name=name+"_block3conv2")
    # Fourth Block (V)
    b4l1 = conv3d_bn_relu(b3l2, is_training, 3, 512, stride=2, name=name+"_block4conv1")
    b4l2 = conv3d_bn_relu(b4l1, is_training, 3, 512, name=name+"_block4conv2")
    seg4 = deconv3d_w_bias(b4l2, 8, 1, batch_size, name=name+"_fourth_deconv")
    # Fifth Block (^)
    b5l1 = deconv3d_w_bias(b4l2, 2, 256, batch_size, name=name+"_block5conv1")
    b5l1 = batch_norm(b5l1, is_training, name=name+"_block5conv1_bn")
    b5l1 = tf.maximum(b5l1, 0.0)
    b5l1 = tf.concat([b5l1,b3l2], 4)
    b5l2 = conv3d_bn_relu(b5l1, is_training, 3, 256, name=name+"_block5conv2")
    seg3 = deconv3d_w_bias(b5l2, 4, 1, batch_size, name=name+"_third_deconv")
    # Sixth Block (^)
    b6l1 = deconv3d_w_bias(b5l2, 2, 128, batch_size, name=name+"_block6conv1")
    b6l1 = batch_norm(b6l1, is_training, name=name+"_block6conv1_bn")
    b6l1 = tf.maximum(b6l1, 0.0)
    b6l1 = tf.concat([b6l1,b2l2], 4)
    b6l2 = conv3d_bn_relu(b6l1, is_training, 3, 128, name=name+"_block6conv2")
    seg2 = deconv3d_w_bias(b6l2, 2, 1, batch_size, name=name+"_second_deconv")
    # Seventh Block (^)
    b7l1 = deconv3d_w_bias(b6l2, 2, 64, batch_size, name=name+"_block7conv1")
    b7l1 = batch_norm(b7l1, is_training, name=name+"_block7conv1_bn")
    b7l1 = tf.maximum(b7l1, 0.0)
    b7l1 = tf.concat([b7l1, b1l2], 4)
    b7l2 = conv3d_bn_relu(b7l1, is_training, 3, 64, name=name+"_block7conv2")
    seg1 = deconv3d_w_bias(b7l2, 1, 1, batch_size, name=name+"_first_deconv")
    return seg1,seg2,seg3,seg4

def chestCT_Net(layer, is_training, keep_prob, class_num, batch_size, name="chestCT_Net"):
    seg1 = 0
    seg2 = 0
    seg3 = 0
    seg4 = 0
    # First Layer
    layer = conv3d_bn_relu(layer, is_training, 7, 64, name=name+"_conv1")
    # First ResNet Block
    layer_res = conv3d_wo_bias(layer, 3, 64, stride=1, name=name+"_res1a")
    layer_res = bn_relu_conv3d(layer_res, is_training, 3, 64, name=name+"_res1b")
    layer = layer + layer_res
    # Second ResNet Block
    layer_res = bn_relu_conv3d(layer, is_training, 3, 64, name=name+"_res2a")
    layer_res = bn_relu_conv3d(layer_res, is_training, 3, 64, name=name+"_res2b")
    layer = layer + layer_res
    # Third ResNet Block
    layer_res = bn_relu_conv3d(layer, is_training, 3, 64, name=name+"_res3a")
    layer_res = bn_relu_conv3d(layer_res, is_training, 3, 64, name=name+"_res3b")
    layer = layer + layer_res
    # First output
    seg1   += deconv3d_w_bias(layer, 1, 1, batch_size, name=name+"_first_deconv")
    # Fourth ResNet Block
    layer_res = bn_relu_conv3d(layer, is_training, 3, 128, stride=2, name=name+"_res4a")
    layer_res = bn_relu_conv3d(layer_res, is_training, 3, 128, name=name+"_res4b")
    layer = tf.nn.avg_pool3d(layer, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='VALID')
    layer = tf.pad(layer, [[0,0],[0,0],[0,0],[0,0],[0,64]])
    layer = layer + layer_res
    # Fifth ResNet Block
    layer_res = bn_relu_conv3d(layer, is_training, 3, 128, name=name+"_res5a")
    layer_res = bn_relu_conv3d(layer_res, is_training, 3, 128, name=name+"_res5b")
    layer = layer + layer_res
    # Sixth ResNet Block
    layer_res = bn_relu_conv3d(layer, is_training, 3, 128, name=name+"_res6a")
    layer_res = bn_relu_conv3d(layer_res, is_training, 3, 128, name=name+"_res6b")
    layer = layer + layer_res
    # Seventh Res Net Block
    layer_res = bn_relu_conv3d(layer, is_training, 3, 128, name=name+"_res7a")
    layer_res = bn_relu_conv3d(layer_res, is_training, 3, 128, name=name+"_res7b")
    layer = layer + layer_res
    # Second output
    seg2   += deconv3d_w_bias(layer, 2, 1, batch_size, name=name+"_second_deconv")
    # Eigth ResNet Block
    layer_res = bn_relu_conv3d(layer, is_training, 3, 256, stride=2, name=name+"_res8a")
    layer_res = bn_relu_conv3d(layer_res, is_training, 3, 256, name=name+"_res8b")
    layer = tf.nn.avg_pool3d(layer, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='VALID')
    layer = tf.pad(layer, [[0,0],[0,0],[0,0],[0,0],[0,128]])
    layer = layer + layer_res
    # Ninth ResNet Block
    layer_res = bn_relu_conv3d(layer, is_training, 3, 256, name=name+"_res9a")
    layer_res = bn_relu_conv3d(layer_res, is_training, 3, 256, name=name+"_res9b")
    layer = layer + layer_res
    # Ath ResNet Block
    layer_res = bn_relu_conv3d(layer, is_training, 3, 256, name=name+"_resAa")
    layer_res = bn_relu_conv3d(layer_res, is_training, 3, 256, name=name+"_resAb")
    layer = layer + layer_res
    # Bth ResNet Block
    layer_res = bn_relu_conv3d(layer, is_training, 3, 256, name=name+"_resBa")
    layer_res = bn_relu_conv3d(layer_res, is_training, 3, 256, name=name+"_resBb")
    layer = layer + layer_res
    # Cth ResNet Block
    layer_res = bn_relu_conv3d(layer, is_training, 3, 256, name=name+"_resCa")
    layer_res = bn_relu_conv3d(layer_res, is_training, 3, 256, name=name+"_resCb")
    layer = layer + layer_res
    # Dth ResNet Block
    layer_res = bn_relu_conv3d(layer, is_training, 3, 256, name=name+"_resDa")
    layer_res = bn_relu_conv3d(layer_res, is_training, 3, 256, name=name+"_resDb")
    layer = layer + layer_res
    # Third Output
    seg3   += deconv3d_w_bias(layer, 4, 1, batch_size, name=name+"_third_deconv")
    # Eth ResNet Block
    layer_res = bn_relu_conv3d(layer, is_training, 3, 512, stride=2, name=name+"_resEa")
    layer_res = bn_relu_conv3d(layer_res, is_training, 3, 512, name=name+"_resEb")
    layer = tf.nn.avg_pool3d(layer, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='VALID')
    layer = tf.pad(layer, [[0,0],[0,0],[0,0],[0,0],[0,256]])
    layer = layer + layer_res
    # Fth ResNet Block
    layer_res = bn_relu_conv3d(layer, is_training, 3, 512, name=name+"_resFa")
    layer_res = bn_relu_conv3d(layer_res, is_training, 3, 512, name=name+"_resFb")
    layer = layer + layer_res
    # Gth ResNet Block
    layer_res = bn_relu_conv3d(layer, is_training, 3, 512, name=name+"_resGa")
    layer_res = bn_relu_conv3d(layer_res, is_training, 3, 512, name=name+"_resGb")
    layer = layer + layer_res
    # Fourth Output
    seg4   += deconv3d_w_bias(layer, 8, 1, batch_size, name=name+"_fourth_deconv")
    return seg4,seg3,seg2,seg1


def GoogLe_Net(layer, is_training, class_num, batch_size, 
	with_batch_norm=False, name="GoogLe_Net"):
    """
    This is the famous GoogLeNet incarnation of the inception network.
    All the power is in the convs, so this is quite simple.
    INPUTS:
    - layer: (tensor.4d) input tensor.
    - output_size: (int) number of classes we're predicting
    - with_batch_norm: If true, add a batch norm to every layer
    - name: (str) the name of the network
    """
    seg = 0
    seg0 = 0
    temp = 0
    seg1 = 0
    seg2 = 0
    seg3 = 0
    m = layer.get_shape().as_list()[1]
    ####layer = tf.py_func(DoG_filters, [layer], [tf.float32])[0]
    ####layer = tf.reshape(layer, [batch_size, m,m,m, 9])
    # Conv1
    layer = conv3d_bn_relu(layer, is_training, 7, 64, stride=2, with_batch_norm=with_batch_norm, name=name+"_conv1")#64->32
    #layer = max_pool3d(layer, k=3, stride=2)
    # Conv2
    layer = conv3d_bn_relu(layer, is_training, 1, 64, with_batch_norm=with_batch_norm, name=name+"_conv2a")#64->32
    layer = conv3d_bn_relu(layer, is_training, 3, 192, with_batch_norm=with_batch_norm, name=name+"_conv2b")#192->96
    bs,m,n,s,c = layer.get_shape().as_list()
    seg1   += deconv3d_w_bias(layer, 2, class_num, batch_size, name=name+"_incept2b_deconv")
    layer = max_pool3d(layer, k=3, stride=2)
    # Incept3
    layer = inceptionv1_module(layer, is_training, kSize=[64,96,128,16,32,32], with_batch_norm=with_batch_norm, name=name+"_incept3a") # 368 total
    bs,m,n,s,c = layer.get_shape().as_list()
    layer = inceptionv1_module(layer, is_training, kSize=[128,128,192,32,96,64], with_batch_norm=with_batch_norm, name=name+"_incept3b") # 640 total
    bs,m,n,s,c = layer.get_shape().as_list()
    #layer = max_pool3d(layer, k=3, stride=2)
    # Incept4
    layer = inceptionv1_module(layer, is_training, kSize=[192,96,208,16,48,64], with_batch_norm=with_batch_norm, name=name+"_incept4a") # 624 total
    bs,m,n,s,c = layer.get_shape().as_list()
    seg2   += deconv3d_w_bias(layer, 4, class_num, batch_size, name=name+"_incept4b_deconv")
    layer = inceptionv1_module(layer, is_training, kSize=[128,128,256,24,64,64], with_batch_norm=with_batch_norm, name=name+"_incept4c")
    bs,m,n,s,c = layer.get_shape().as_list()
    layer = inceptionv1_module(layer, is_training, kSize=[112,144,288,32,64,64], with_batch_norm=with_batch_norm, name=name+"_incept4d")
    bs,m,n,s,c = layer.get_shape().as_list()
    layer = inceptionv1_module(layer, is_training, kSize=[256,160,320,32,128,128], with_batch_norm=with_batch_norm, name=name+"_incept4e")
    bs,m,n,s,c = layer.get_shape().as_list()
    layer = max_pool3d(layer, k=3, stride=2)
    # Incept5
    layer = inceptionv1_module(layer, is_training, kSize=[256,160,320,32,128,128], with_batch_norm=with_batch_norm, name=name+"_incept5a")
    bs,m,n,s,c = layer.get_shape().as_list()
    layer = inceptionv1_module(layer, is_training, kSize=[384,192,384,48,128,128], with_batch_norm=with_batch_norm, name=name+"_incept5b")
    bs,m,n,s,c = layer.get_shape().as_list()
    seg3   += deconv3d_w_bias(layer, 8, class_num, batch_size, name=name+"_incept5b_deconv")
    #return seg3 + seg2 + seg1, seg3,seg2,seg1#seg,reg,cls#cls#seg + 1e-12
    #return seg3,seg2,seg1#seg,reg,cls#cls#seg + 1e-12
    return (seg3,)

def U_Net_chunk(layer, is_training, num_layers, num_filters, name='', 
        with_batch_norm=False):
    """
        Create a 'chunk' consisting of two 3x3x3 convolution layers.
    """
    for i in range(num_layers):
        layer = conv3d_bn_relu(layer, is_training, 3, num_filters, stride=1, 
            with_batch_norm=with_batch_norm, 
            name=name + "_conv%d" % i)

    return layer

def U_Net(layer, is_training, num_class, batch_size, name="U_Net", with_batch_norm=False):

    # Number of filters per chunk
    #layer_depths = [64, 128, 256, 512]
    layer_depths = [32, 64, 64, 128]
    layers_per_chunk = 2

    # Make the encoder
    encode_feats = []
    for idx, depth in enumerate(layer_depths):
        # Conv-net chunk
        layer = U_Net_chunk(layer, is_training, layers_per_chunk, depth, 
                name=name+"layer_%d" % idx, 
                with_batch_norm=with_batch_norm)

        # Store features and pool for all but the last chunk
        if idx < len(layer_depths) - 1:
            encode_feats.append(layer)
            layer = max_pool3d(layer, k=3, stride=2)

    # Make the decoder
    for idx, depth in enumerate(reversed(layer_depths[:-1])):
        # Upsample (deconv)
        decode_idx = idx + len(layer_depths)
        layer = deconv3d_w_bias(layer, 2, depth, batch_size, 
                name=name+"deconv_%d" % idx)

        # Concatenate the encoder features
        layer = tf.concat((layer, encode_feats[-1 - idx]), axis=-1)

        # Conv-net chunk
        layer = U_Net_chunk(layer, is_training, layers_per_chunk, depth, 
                name=name+"layer_%d" % decode_idx, 
                with_batch_norm=with_batch_norm)

    # Final classifier layer
    layer = conv3d_w_bias(layer, 1, num_class, name=name+'_conv_final')

    return (layer,)

def Inception_Net(layer, is_training, class_num, batch_size, name="Inceptionv3_Net"):
    """
    This is the famous Inception v3 Network.
    This is a big big fucking network.
    INPUTS:
    - layer: (tensor.4d) input tensor.
    - output_size: (int) number of classes we're predicting
    - keep_prob: (float) probability to keep during dropout.
    - name: (str) the name of the network
    """
    # 224x224x?
    layer = conv2d_bn_relu(layer, is_training, 3, 32, stride=2, name=name+'_conv0')
    # 112x112x32
    layer = conv2d_bn_relu(layer, is_training, 3, 32, name=name+'_conv1')
    # 112x112x32
    layer = conv2d_bn_relu(layer, is_training, 3, 64, name=name+'_conv2')
    layer = max_pool(layer, k=3, stride=2)
    # 56x56x64
    layer = conv2d_bn_relu(layer, is_training, 1, 80, name=name+'_conv3')
    # 56x56x80
    layer = conv2d_bn_relu(layer, is_training, 3, 192, name=name+'_conv4')
    layer = max_pool(layer, k=3, stride=2)
    # 28x28x192
    branch1x1 = conv2d_bn_relu(layer, is_training, 1, 64, name=name+'_incept1branch1')
    branch5x5 = conv2d_bn_relu(layer, is_training, 1, 48, name=name+'_incept1branch5a')
    branch5x5 = conv2d_bn_relu(branch5x5, is_training, 5, 64, name=name+'_incept1branch5b')
    branch3x3 = conv2d_bn_relu(layer, is_training, 1, 64, name=name+'_incept1branch3a')
    branch3x3 = conv2d_bn_relu(branch3x3, is_training, 3, 96, name=name+'_incept1branch3b')
    branch3x3 = conv2d_bn_relu(branch3x3, is_training, 3, 96, name=name+'_incept1branch3c')
    branchpool = tf.nn.avg_pool(layer, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
    branchpool = conv2d_bn_relu(branchpool, is_training, 1, 32, name=name+'_incept1branchpool')
    layer = tf.concat([branch1x1, branch5x5, branch3x3, branchpool], 3)
    # 28x28x256
    branch1x1 = conv2d_bn_relu(layer, is_training, 1, 64, name=name+'_incept2branch1')
    branch5x5 = conv2d_bn_relu(layer, is_training, 1, 48, name=name+'_incept2branch5a')
    branch5x5 = conv2d_bn_relu(branch5x5, is_training, 5, 64, name=name+'_incept2branch5b')
    branch3x3 = conv2d_bn_relu(layer, is_training, 1, 64, name=name+'_incept2branch3a')
    branch3x3 = conv2d_bn_relu(branch3x3, is_training, 3, 96, name=name+'_incept2branch3b')
    branch3x3 = conv2d_bn_relu(branch3x3, is_training, 3, 96, name=name+'_incept2branch3c')
    branchpool = tf.nn.avg_pool(layer, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
    branchpool = conv2d_bn_relu(branchpool, is_training, 1, 64, name=name+'_incept2branchpool')
    layer = tf.concat([branch1x1, branch5x5, branch3x3, branchpool], 3)
    # 28x28x288
    branch1x1 = conv2d_bn_relu(layer, is_training, 1, 64, name=name+'_incept3branch1')
    branch5x5 = conv2d_bn_relu(layer, is_training, 1, 48, name=name+'_incept3branch5a')
    branch5x5 = conv2d_bn_relu(branch5x5, is_training, 5, 64, name=name+'_incept3branch5b')
    branch3x3 = conv2d_bn_relu(layer, is_training, 1, 64, name=name+'_incept3branch3a')
    branch3x3 = conv2d_bn_relu(branch3x3, is_training, 3, 96, name=name+'_incept3branch3b')
    branch3x3 = conv2d_bn_relu(branch3x3, is_training, 3, 96, name=name+'_incept3branch3c')
    branchpool = tf.nn.avg_pool(layer, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
    branchpool = conv2d_bn_relu(branchpool, is_training, 1, 64, name=name+'_incept3branchpool')
    layer = tf.concat([branch1x1, branch5x5, branch3x3, branchpool], 3)
    # 28x28x288
    branch1x1 = conv2d_bn_relu(layer, is_training, 3, 384, stride=2, name=name+'_incept4branch1')
    branch3x3 = conv2d_bn_relu(layer, is_training, 1, 64, name=name+'_incept4branch3a')
    branch3x3 = conv2d_bn_relu(branch3x3, is_training, 3, 96, name=name+'_incept4branch3b')
    branch3x3 = conv2d_bn_relu(branch3x3, is_training, 3, 96, stride=2, name=name+'_incept4branch3c')
    branchpool = max_pool(layer, k=3, stride=2)
    layer = tf.concat([branch1x1, branch3x3, branchpool], 3)
    # 14x14x768
    branch1 = conv2d_bn_relu(layer, is_training, 1, 192, name=name+'_incept5branch1')
    branch7a = conv2d_bn_relu(layer, is_training, 1, 128, name=name+'_incept5branch7Aa')
    branch7a = conv2d_bn_relu(branch7a, is_training, [1, 7], 128, name=name+'_incept5branch7Ab')
    branch7a = conv2d_bn_relu(branch7a, is_training, [7, 1], 192, name=name+'_incept5branch7Ac')
    branch7b = conv2d_bn_relu(layer, is_training, 1, 128, name=name+'_incept5branch7Ba')
    branch7b = conv2d_bn_relu(branch7b, is_training, [7, 1], 128, name=name+'_incept5branch7Bb')
    branch7b = conv2d_bn_relu(branch7b, is_training, [1, 7], 128, name=name+'_incept5branch7Bc')
    branch7b = conv2d_bn_relu(branch7b, is_training, [7, 1], 128, name=name+'_incept5branch7Bd')
    branch7b = conv2d_bn_relu(branch7b, is_training, [1, 7], 192, name=name+'_incept5branch7Be')
    branchpool = tf.nn.avg_pool(layer, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
    branchpool = conv2d_bn_relu(branchpool, is_training, 1, 192, name=name+'_incept5branchpool')
    layer = tf.concat([branch1, branch7a, branch7b, branchpool], 3)
    # 14x14x768
    branch1 = conv2d_bn_relu(layer, is_training, 1, 192, name=name+'_incept6branch1')
    branch7a = conv2d_bn_relu(layer, is_training, 1, 160, name=name+'_incept6branch7Aa')
    branch7a = conv2d_bn_relu(branch7a, is_training, [1, 7], 160, name=name+'_incept6branch7Ab')
    branch7a = conv2d_bn_relu(branch7a, is_training, [7, 1], 192, name=name+'_incept6branch7Ac')
    branch7b = conv2d_bn_relu(layer, is_training, 1, 128, name=name+'_incept6branch7Ba')
    branch7b = conv2d_bn_relu(branch7b, is_training, [7, 1], 160, name=name+'_incept6branch7Bb')
    branch7b = conv2d_bn_relu(branch7b, is_training, [1, 7], 160, name=name+'_incept6branch7Bc')
    branch7b = conv2d_bn_relu(branch7b, is_training, [7, 1], 160, name=name+'_incept6branch7Bd')
    branch7b = conv2d_bn_relu(branch7b, is_training, [1, 7], 192, name=name+'_incept6branch7Be')
    branchpool = tf.nn.avg_pool(layer, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
    branchpool = conv2d_bn_relu(branchpool, is_training, 1, 192, name=name+'_incept6branchpool')
    layer = tf.concat([branch1, branch7a, branch7b, branchpool], 3)
    seg   = deconv2d_wo_bias(layer, 16, class_num, batch_size, name=name+"_incept6_deconv")
    # 14x14x768
    branch1 = conv2d_bn_relu(layer, is_training, 1, 192, name=name+'_incept7branch1')
    branch7a = conv2d_bn_relu(layer, is_training, 1, 160, name=name+'_incept7branch7Aa')
    branch7a = conv2d_bn_relu(branch7a, is_training, [1, 7], 160, name=name+'_incept7branch7Ab')
    branch7a = conv2d_bn_relu(branch7a, is_training, [7, 1], 192, name=name+'_incept7branch7Ac')
    branch7b = conv2d_bn_relu(layer, is_training, 1, 128, name=name+'_incept7branch7Ba')
    branch7b = conv2d_bn_relu(branch7b, is_training, [7, 1], 160, name=name+'_incept7branch7Bb')
    branch7b = conv2d_bn_relu(branch7b, is_training, [1, 7], 160, name=name+'_incept7branch7Bc')
    branch7b = conv2d_bn_relu(branch7b, is_training, [7, 1], 160, name=name+'_incept7branch7Bd')
    branch7b = conv2d_bn_relu(branch7b, is_training, [1, 7], 192, name=name+'_incept7branch7Be')
    branchpool = tf.nn.avg_pool(layer, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
    branchpool = conv2d_bn_relu(branchpool, is_training, 1, 192, name=name+'_incept7branchpool')
    layer = tf.concat([branch1, branch7a, branch7b, branchpool], 3)
    # 14x14x768
    branch1 = conv2d_bn_relu(layer, is_training, 1, 192, name=name+'_incept8branch1')
    branch7a = conv2d_bn_relu(layer, is_training, 1, 192, name=name+'_incept8branch7Aa')
    branch7a = conv2d_bn_relu(branch7a, is_training, [1, 7], 192, name=name+'_incept8branch7Ab')
    branch7a = conv2d_bn_relu(branch7a, is_training, [7, 1], 192, name=name+'_incept8branch7Ac')
    branch7b = conv2d_bn_relu(layer, is_training, 1, 128, name=name+'_incept8branch7Ba')
    branch7b = conv2d_bn_relu(branch7b, is_training, [7, 1], 192, name=name+'_incept8branch7Bb')
    branch7b = conv2d_bn_relu(branch7b, is_training, [1, 7], 192, name=name+'_incept8branch7Bc')
    branch7b = conv2d_bn_relu(branch7b, is_training, [7, 1], 192, name=name+'_incept8branch7Bd')
    branch7b = conv2d_bn_relu(branch7b, is_training, [1, 7], 192, name=name+'_incept8branch7Be')
    branchpool = tf.nn.avg_pool(layer, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
    branchpool = conv2d_bn_relu(branchpool, is_training, 1, 192, name=name+'_incept8branchpool')
    layer = tf.concat([branch1, branch7a, branch7b, branchpool], 3)
    # 14x14x768
    branch3x3 = conv2d_bn_relu(layer, is_training, 1, 192, name=name+'_incept9branch3a')
    branch3x3 = conv2d_bn_relu(branch3x3, is_training, 3, 320, stride=2, name=name+'_incept9branch3b')
    branch7x7 = conv2d_bn_relu(layer, is_training, 1, 192, name=name+'_incept9branch7a')
    branch7x7 = conv2d_bn_relu(branch7x7, is_training, [1, 7], 192, name=name+'_incept9branch7b')
    branch7x7 = conv2d_bn_relu(branch7x7, is_training, [7, 1], 192, name=name+'_incept9branch7c')
    branch7x7 = conv2d_bn_relu(branch7x7, is_training, 3, 192, stride=2, name=name+'_incept9branch7d')
    branchpool = max_pool(layer, k=3, stride=2)
    layer = tf.concat([branch3x3, branch7x7, branchpool], 3)
    # 7x7x1280
    branch1 = conv2d_bn_relu(layer, is_training, 1, 320, name=name+'_inceptAbranch1')
    branch3a = conv2d_bn_relu(layer, is_training, 1, 384, name=name+'_inceptAbranch3Aa')
    branch3a = tf.concat([conv2d_bn_relu(branch3a, is_training, [1, 3], 384, name=name+'_inceptAbranch3Ab'),
                             conv2d_bn_relu(branch3a, is_training, [3, 1], 384, name=name+'_inceptAbranch3Ac')], 3)
    branch3b = conv2d_bn_relu(layer, is_training, 1, 448, name=name+'_inceptAbranch3Ba')
    branch3b = conv2d_bn_relu(branch3b, is_training, 3, 384, name=name+'_inceptAbranch3Bb')
    branch3b = tf.concat([conv2d_bn_relu(branch3b, is_training, [1, 3], 384, name=name+'_inceptAbranch3Bc'),
                             conv2d_bn_relu(branch3b, is_training, [3, 1], 384, name=name+'_inceptAbranch3Bd')], 3)
    branchpool = tf.nn.avg_pool(layer, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
    branchpool = conv2d_bn_relu(branchpool, is_training, 1, 192, name=name+'_inceptAbranchpool')
    layer = tf.concat([branch1, branch3a, branch3b, branchpool], 3)
    # 7x7x2048
    branch1 = conv2d_bn_relu(layer, is_training, 1, 320, name=name+'_inceptBbranch1')
    branch3a = conv2d_bn_relu(layer, is_training, 1, 384, name=name+'_inceptBbranch3Aa')
    branch3a = tf.concat([conv2d_bn_relu(branch3a, is_training, [1, 3], 384, name=name+'_inceptBbranch3Ab'),
                             conv2d_bn_relu(branch3a, is_training, [3, 1], 384, name=name+'_inceptBbranch3Ac')], 3)
    branch3b = conv2d_bn_relu(layer, is_training, 1, 448, name=name+'_inceptBbranch3Ba')
    branch3b = conv2d_bn_relu(branch3b, is_training, 3, 384, name=name+'_inceptBbranch3Bb')
    branch3b = tf.concat([conv2d_bn_relu(branch3b, is_training, [1, 3], 384, name=name+'_inceptBbranch3Bc'),
                             conv2d_bn_relu(branch3b, is_training, [3, 1], 384, name=name+'_inceptBbranch3Bd')], 3)
    branchpool = tf.nn.avg_pool(layer, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
    branchpool = conv2d_bn_relu(branchpool, is_training, 1, 192, name=name+'_inceptBbranchpool')
    layer = tf.concat([branch1, branch3a, branch3b, branchpool], 3)
    seg   += deconv2d_w_bias(layer, 32, class_num, batch_size, name=name+"_inceptB_deconv")
    return seg + 1e-12

def conv_res(layer, is_training, architecture=[[1, 64], [3, 64], [1, 256]], alpha=0.1, name="conv_res"):
    """
    This is going to be a residual layer.
    We do 3 convolutions and add to the original input.
    INPUTS:
    - layer: (tensor.4d) input tensor
    - is_training: (variable) whether or not we're training
    - architecture: (list of lists) architecture of 3 convs
    - alpha: (float) for the relu
    - name: (string) name of the layer
    """
    l_input = layer #save for later
    for iter_num, kSize in enumerate(architecture):
        layer = batch_norm(layer, is_training, name=(name+'_bn'+str(iter_num)))
        layer = tf.maximum(layer, layer*alpha)
        layer = conv2d_wo_bias(layer, kSize[0], kSize[1], name=(name+"_conv2d"+str(iter_num)))
    if l_input.get_shape().as_list()[3] != kSize[1]:
        l_input = tf.pad(l_input, [[0,0],[0,0],[0,0],[0,kSize[1]-l_input.get_shape().as_list()[3]]])
    layer += l_input
    return layer

def Res_Net(layer, is_training, class_num, batch_size, name="Res_Net"):
    """
    This is the famous Res Net.
    150+ Layers mother fucker!  Fuck that shit..
    INPUTS:
    - layer: (tensor.4d) input tensor.
    - output_size: (int) number of classes we're predicting
    - keep_prob: (float) probability to keep during dropout.
    - name: (str) the name of the network
    """
    layer = conv2d_wo_bias(layer, 7, 64, stride=2, name=name+"_conv1")
    layer = max_pool(layer, k=3, stride=2)
    for i in range(3):
        layer = conv_res(layer, is_training, architecture=[[1,64],[3,64],[1,256]], name=name+"_conv2_"+str(i))
    layer = max_pool(layer, k=3, stride=2)
    for i in range(8):
        layer = conv_res(layer, is_training, architecture=[[1,128],[3,128],[1,512]], name=name+"_conv3_"+str(i))
    layer = max_pool(layer, k=3, stride=2)
    seg   = deconv2d_wo_bias(layer, 16, class_num, batch_size, name=name+"_covn4_deconv")
    for i in range(36):
        layer = conv_res(layer, is_training, architecture=[[1,256],[3,256],[1,1024]], name=name+"_conv4_"+str(i))
    layer = max_pool(layer, k=3, stride=2)
    for i in range(3):
        layer = conv_res(layer, is_training, architecture=[[1,512],[3,512],[1,2048]], name=name+"_conv5_"+str(i))
    seg   += deconv2d_w_bias(layer, 32, class_num, batch_size, name=name+"_conv5_deconv")
    return seg
