import matplotlib.pyplot as plt
import h5py
import numpy as np
import os
from os import listdir, remove, mkdir
from os.path import isfile, join, isdir
import scipy
import scipy.misc
import scipy.ndimage
from tensorflow.python.platform import gfile
import socket
import sys
import time
import datetime
import logging
import traceback
import pickle        

# Compatibility with TF 2.0--this is TF 1 code
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except:
    import tensorflow as tf

from pyCudaImageWarp import augment3d 

from .layers import *
from .nets_segmentation3d import *
from .CTH_seg_inference.CTH_seg_common.data import *
from .ops import *

from .CTH_seg_inference import inference

# Codes for different loss types
perObjectNoneCode = 0
perObjectVoronoiCode = 1
perObjectFuzzyCode = 2
fuzzyNumVertices = 4

"""
    Freeze the graph for later use. Does not require class instantiation
"""
def freeze_graph(sess, pb_path):

    # Freeze the graph, with 'prob' as the only output
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        ['prob'])

    # Save the frozen graph
    with open(pb_path, 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())

"""
    Compute the mean and sum of squared differences online, using Welford's algorithm.
"""
def __update_mean_var__(mean, sumSquares, count, newVal):

    if count < 2:
        return (newVal, 0, 2)

    newCount = count + 1
    delta = newVal - mean
    newMean = mean + delta / newCount
    newSumSquares = sumSquares + delta * (newVal - newMean)

    return (newMean, newSumSquares, newCount)

"""
Get the shape of a tensor
"""
def __tensor_shape__(x):
    return tuple([int(k) for k in x.shape])

"""
    Pad a numpy array (volume) with a channel dimension.
"""
def __pad_channel__(vol):
    while vol.ndim < 4:
        vol = np.expand_dims(vol, -1)
    return vol

"""
    Convert the IOU (Jaccard) score to the Dice-Sorensen index
"""
def iou2Dice(iou):
    return 2 * iou / (1 + iou)  

#TODO finish this
def checkpoint_gradients(cost, grads_and_vars):
    """
    Add checkpoints to the existing grads_and_vars, for memory efficiency.
    Uses the collection 'checkpoints' to find the variables for saving. This
    should include all the conv and FC outputs, but not the BN, ReLu, etc.
    """
    from .memory_saving_gradients import gradients

    _, auto_vars = zip(*grads_and_vars) # Something like this to separate into 2 lists
    checkpoint_grads = gradients(cost, list(auto_vars), checkpoints='collection')
    return zip(checkpoint_grads, auto_vars) # Same format as grads_and_vars
    
def average_gradients(grads_multi):
    """
    Basically averages the aggregated gradients.
    Much was stolen from code from the Tensorflow team.
    Basically, look at the famous inceptionv3 code.
    INPUTS:
    - grads_multi: a list of gradients and variables
    """
    average_grads = []
    for grad_and_vars in zip(*grads_multi):
        grads = []
        for g,_ in grad_and_vars:
            if g is None:
                continue
            expanded_g = tf.expand_dims(g,0)
            grads.append(expanded_g)
        if grads == []:
            continue
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

class segmentor:
    def __init__(self, opts):
        """
        Initialization of all the fields.
        We also create the network.
        INPUTS:
        - opts: (object) command line arguments from argparser
        """

        # Assume we have a display unless proved otherwise
        self.have_display = True

        # Parse the 'train_gpus' argument into integers
        if opts.train_gpus is None:
            raise ValueError('Must supply train_gpus option')
        opts.num_splits = len(opts.train_gpus)

        # Get the augmentation mode--GPU or CPU
        if opts.augment_gpu is None:
            opts.augment_api = 'scipy' 
            print("Using CPU data augmentation")
        else:
            opts.augment_api = 'cuda' 
            print("Using GPU data augmentation on device %d" % opts.augment_gpu)

        # Set the model parameters needed for inference
        self.params = {
            'num_class': opts.num_class
        }
        tf.reset_default_graph()

        # Get the loss function
        lossFunMap = {
                'softmax': {
                    'single': get_softmax_loss,
                    'batch': get_softmax_loss
                },
                'iou': {
                    'single': get_iou_loss,
                    'batch': get_iou_loss_batch,
                }
        }
        lossFun = lossFunMap[opts.loss_type]

        # Save the options
        self.opts = opts

        # Print stuff about the options
        self.super_print("Training GPUs: ")
        self.super_print(opts.train_gpus)

        # Disable balanced loss for certain types
        allowBalanced = {
                'softmax': True,
                'iou': False
        }
        if opts.balanced and not allowBalanced[opts.loss_type]:
            self.super_print('Warning: disabling balanced loss for type ' + \
                    opts.loss_type)
            opts.balanced = False

        # Get the per-object loss type
        perObjectMap = {
            'none' : perObjectNoneCode,
            'voronoi' : perObjectVoronoiCode,
            'fuzzy' : perObjectFuzzyCode
        }
        self.per_object_code = perObjectMap[opts.per_object]
        self.have_per_object = self.per_object_code > 0

        # Disable per-object loss for certain types
        allowPerObject = {
            'softmax': False,
            'iou': True
        }
        if self.have_per_object and not allowPerObject[opts.loss_type]:
            self.super_print('Warning: disabling per-object loss for type ' + \
                    opts.loss_type)
            self.opts.per_object = 'none'
            self.per_object_code = perObjectNoneCode
            self.have_per_object = False

        # Disable background loss for certain types
        allowBackgroundLoss = {
            'softmax': False,
            'iou': True
        }
        if opts.background_loss and not allowBackgroundLoss[opts.loss_type]:
            self.super_print('Warning: disabling background loss for type ' + \
                    opts.loss_type)
            self.opts.background_loss = False
            opts.background_loss = False


        # Interpret the network type string
        networkTypeMap = {
                'GoogLe': GoogLe_Net,
                'uNet': U_Net
        }
        try:
            networkFun = networkTypeMap[opts.network]
        except KeyError:
            raise ValueError('Unrecognized network type: ' + opts.network)

        # Verify the saving criterion
        if opts.criterion == 'val': 
            if not opts.validate:
                raise ValueError('Cannot have criterion ' + opts.criterion  
                    + ' with validation set to ' + boolToYesNo[opts.validate])
        elif not opts.criterion == 'train':
            raise ValueError('Unrecognized criterion: ' + opts.criterion)
        
        # Print basic options
        yesStr = 'yes'
        noStr = 'no'
        boolToYesNo = {1: yesStr, True: yesStr, 0: noStr, False: noStr}
        self.super_print('Model type: ' + opts.network)
        self.super_print('Using batch norm: ' + boolToYesNo[opts.batch_norm])
        self.super_print('Loss type: ' + opts.loss_type)
        self.super_print('Background loss: ' + boolToYesNo[opts.background_loss])
        self.super_print('Using balanced loss: ' + boolToYesNo[opts.balanced])
        self.super_print("Minimum object size for detection: %d" % \
                opts.min_object_size)
        self.super_print("Out-of-bounds label: %d" % opts.oob_label)
        self.super_print('Out-of-bounds image value: ' + str(opts.oob_image_val))
        self.super_print('Using augmentation: ' + boolToYesNo[opts.augment])
        self.super_print("Iteration of maximum augmentation: %d" % opts.max_aug_iter)
        self.super_print('Using a mask: ' + boolToYesNo[opts.masked])
        self.super_print('Running validation: ' + boolToYesNo[opts.validate])
        self.super_print('Saving criterion: ' + opts.criterion)
        self.super_print('Freezing model at each epoch: ' + boolToYesNo[opts.save_frozen])

        # Decide whether to use the CuDNN auto-tuner
        self.super_print('Using autotune: ' + boolToYesNo[opts.autotune])
        os.environ['TF_CUDNN_USE_AUTOTUNE'] = str(opts.autotune)

        # Noise levels
        ctNoise = 5
        petNoise = 0.25

        # Map window types to HU range
        windowMap = {'liver' : {
                        'window_min': np.array([-50]),
                        'window_max': np.array([250]),
                        'noise': np.array([ctNoise])
                     },
                     'all' : {
                         'window_min': np.array([-200]), 
                         'window_max': np.array([400]),
                         'noise': np.array([ctNoise])
                     },
                     'pet-ct-lung': {
                         'window_min': np.array([-float('inf'), -1350]),
                         'window_max': np.array([float('inf'), 400]),
                         'noise': np.array([petNoise, ctNoise])
                     },
                     'pet-ct-all': {
                         'window_min': np.array([-float('inf'), -200]),
                         'window_max': np.array([float('inf'), 400]),
                         'noise': np.array([petNoise, ctNoise])
                 }
         }

        # Get the window range
        if opts.window is None:
                self.params['window_min'] = np.array([-float('inf') for x in range(num_channels)])
                self.params['window_max'] = np.array([float('inf') for x in range(num_channels)])
                self.opts.noiseLevel = 0
        else:
                windowParams = windowMap[opts.window]
                self.params['window_min'] = windowParams['window_min']
                self.params['window_max'] = windowParams['window_max']
                self.opts.noiseLevel = windowParams['noise']
        self.super_print("Using window range %s to %s (%s)" % \
                (str(self.params['window_min']), str(self.params['window_max']), 
                opts.window)
        )
        if self.opts.augment:
            self.super_print('Using noise level ' + str(self.opts.noiseLevel))

        # Read the size of a data element
        if self.opts.path_train:
            self.params['label_shape'], self.num_channels = find_data_shape(self.opts.path_train)
        elif self.opts.path_test:
            self.params['label_shape'], self.num_channels = find_data_shape(self.opts.path_test)
        elif self.opts.path_inference:
            self.params['label_shape'], self.num_channels = find_data_shape(self.opts.path_inference)
        else:
            raise ValueError('Failed to infer the input size!')
        self.super_print("Inferred %d channel(s) from the input" % \
                self.num_channels)

        # Override the matrix size, if it was given
        if opts.crop_size is not None:
            if np.equal(opts.crop_size, self.params['label_shape'][:3]).all():
                self.super_print("""Inferred image shape matches provided shape %s""" % \
                (opts.crop_size,))
            else:
                self.super_print("Overriding image shape of %s with shape %s" % \
                    (self.params['label_shape'][:3], opts.crop_size))
                new_shape = tuple(opts.crop_size)
                if len(self.params['label_shape']) > len(new_shape):
                    new_shape = (new_shape 
                        + self.params['label_shape'][len(new_shape) + 1:])
                self.params['label_shape'] = tuple(opts.crop_size)
        else:
            self.super_print("Using inferred image shape of %s" % \
                (self.params['label_shape'],))

        # Get the data shape
        self.params['data_shape'] = self.params['label_shape'] + (self.num_channels,)

        # Create the sizes
        xTe_size = (1,) + self.params['data_shape']
        yTe_size = (1,) + self.params['label_shape']
        xTr_size = (opts.batch_size,) + self.params['data_shape']
        yTr_size = (opts.batch_size,) + self.params['label_shape']

        # Make the basic placeholders
        self.X = {'single': tf.placeholder(tf.float32, xTe_size, name='X'),
                'batch': tf.placeholder(tf.float32, xTr_size, name='X_batch')}
        self.Y ={'single': tf.placeholder(tf.int64, yTe_size, name='Y'),
                'batch': tf.placeholder(tf.int64, yTr_size, name='Y_batch')}
        #self.eTe = tf.placeholder(tf.int64, yTe_size)
        #self.eTr = tf.placeholder(tf.int64, yTr_size)

        # Make placeholders for balanced loss
        self.loss_kwargs = {name: {} for name in ['batch', 'single']}
        loss_flags = {}
        if self.opts.balanced:
            self.weight_map = {'batch': tf.placeholder(tf.float32, yTr_size, 
                                   name='weight'),
                               'single': tf.placeholder(tf.float32, yTe_size, 
                                   name='weight_batch')}
            for batchType in ['batch', 'single']:
                self.loss_kwargs[batchType]['weight_map'] = \
                    self.weight_map[batchType]
        else:
            self.weight_map = {'batch' : None, 
                               'single': None}

        # Add kwargs for squished log-probs
        if opts.loss_type == 'iou' and opts.iou_log_prob:
            self.super_print('Using squished log probabilities')
            loss_flags['log_prob'] = True

        # Add kwargs for background loss
        if self.opts.background_loss:
            loss_flags['background_loss'] = True
                
        # Make placeholders for per-object loss
        self.voronoi = {'batch' : None, # Default to None for all of them
                        'single': None}
        self.fuzzy_weights = self.voronoi
        self.fuzzy_labels = self.voronoi

        self.num_objects = {'batch': tf.placeholder(tf.int32, opts.batch_size, 
                name='num_objects_batch'),
            'single': tf.placeholder(tf.int32, 1, name='num_objects')
            } if self.have_per_object else {
                key: None for key in ['batch', 'single']
            }
        if self.have_per_object:
            for batchType in ['batch', 'single']:
                self.loss_kwargs[batchType]['num_objects'] = \
                    self.num_objects[batchType]
        if self.per_object_code == perObjectVoronoiCode:
            self.voronoi = {'batch': tf.placeholder(tf.int32, yTr_size, 
                                name='voronoi'),
                            'single': tf.placeholder(tf.int32, yTe_size, 
                                name='voronoi_batch')}
            for batchType in ['batch', 'single']:
                self.loss_kwargs[batchType]['voronoi'] = self.voronoi[batchType]
        elif self.per_object_code == perObjectFuzzyCode:
            self.fuzzy_vert = 4
            fuzzy_weight_shape = self.params['label_shape'] + (self.fuzzy_vert,)
            self.fuzzy_weights = {
                'batch': tf.placeholder(tf.float32, 
                    (opts.batch_size,) + fuzzy_weight_shape,
                    name='fuzzy_weights_batch'),
                'single': tf.placeholder(tf.float32,
                    (1,) + fuzzy_weight_shape,
                    name='fuzzy_weights_single')
            }
            self.fuzzy_labels = {
                batchType: tf.placeholder(tf.int32, 
                    self.fuzzy_weights[batchType].shape, 
                    name='fuzzy_labels_' + batchType) \
                        for batchType in ['batch', 'single']
            }
            for batchType in ['batch', 'single']:
                self.loss_kwargs[batchType]['fuzzy_weights'] = \
                    self.fuzzy_weights[batchType]
                self.loss_kwargs[batchType]['fuzzy_labels'] = \
                    self.fuzzy_labels[batchType]

        # Listing the data.
        if not self.opts.path_train:
            raise ValueError('Missing the training path!')
        self.train_filenames = listdir(self.opts.path_train)

        # Count the number of voxels in each volume
        sample_probs = np.zeros(len(self.train_filenames))
        for i, filename in enumerate(self.train_filenames):
            shape = read_data_shape(
                    os.path.join(self.opts.path_train, filename)
            )
            sample_probs[i] = np.prod(np.array(shape))

        if opts.sampleProbs:
            # Compute the sampling probability for each volume
            self.super_print('Sampling probabilities for each training item:')
            sample_probs /= np.sum(sample_probs)
            for i, filename in enumerate(self.train_filenames):
                self.super_print(
                        "%s: %f" % (self.train_filenames[i], sample_probs[i])
                )

            # Compute the CDF
            self.sample_cdf = np.cumsum(sample_probs)
        self.super_print('Using sample probabilities: ' + boolToYesNo[opts.sampleProbs])

        # Compute the number of elements per split
        if opts.batch_size % opts.num_splits != 0:
            raise ValueError("""batch size %d is not divisible by number of splits 
                    %d""") % (opts.batch_size, opts.num_splits) 
        data_per_split = int(opts.batch_size / opts.num_splits)

        # Split up the multi-device inputs
        multi_inputs = tf.split(self.X['batch'], opts.num_splits, 0)
        multi_targets = tf.split(self.Y['batch'], opts.num_splits, 0)
        multi_kwargs = {key: tf.split(val, opts.num_splits, 0) \
            for key, val in self.loss_kwargs['batch'].items()}
        multi_kwargs = [{key: val[k] for key, val in multi_kwargs.items()} \
            for k in range(opts.num_splits)]

        # Add in the loss flags to kwargs (put here to folow tf.split())
        for key, val in loss_flags.items():
            self.loss_kwargs['single'][key] = val
            for k in range(opts.num_splits):
                multi_kwargs[k][key] = (val,) * opts.batch_size

        # Creating the Network for Testing
        with tf.variable_scope('network'):
            self.pred = networkFun(self.X['single'], False, opts.num_class, 
                    1, with_batch_norm=opts.batch_norm)
        L2_loss = get_L2_loss(self.opts.l2)
        L1_loss = get_L1_loss(self.opts.l1)
        seg_loss = lossFun['single'](self.pred, self.Y['single'], 
                self.opts.num_class,
                **self.loss_kwargs['single'])
        self.prob = {'single': tf.nn.softmax(self.pred[0], name='prob')}
        self.loss = {'single': {'test': seg_loss}}

        # Set up the optimizer parameters
        optimizer,global_step = get_optimizer(self.opts.lr, self.opts.lr_decay, 
            self.opts.epoch)

        # Instantiate the network on each device
        networkNames = ['train', 'test']
        loss_multi = {name: [] for name in networkNames}
        grads_multi = []
        probs_multi = []
        #tf.get_variable_scope().reuse_variables() #XXX used to have this
        devices = ['/gpu:%d' % i for i in opts.train_gpus] 
        for device, X, Y, kwargs in zip(devices, multi_inputs, 
            multi_targets, multi_kwargs):
            print ("Initializing device %s..." % device)
            with tf.device(device):
                device_name, device_num = device.strip('/').split(':')
                with tf.name_scope(device_name + device_num) as scope:
                    # Instantiate two networks, one for testing and one for training
                    losses = {}
                    for name in networkNames:
                        # Make the network, being sure to reuse everything
                        isTraining = name == 'train'
                        with tf.variable_scope('network', reuse=True):
                            pred = networkFun(X, isTraining, opts.num_class,
                                data_per_split,
                                with_batch_norm=opts.batch_norm)

                        # Get the loss
                        loss = lossFun['batch'](
                            pred, 
                            Y, 
                            opts.num_class, 
                            **kwargs
                        )
                        loss_multi[name].append(loss)

                        # Training-specific stuff
                        if isTraining:
                            # Extract the first image in the batch, for display
                            if devices.index(device) == 0:
                                self.segmentation_example = tf.nn.softmax(pred[0])[0]

                            # Get the total objective function and gradients
                            cost = loss + L2_loss + L1_loss 
                            grads_and_vars = optimizer.compute_gradients(cost)
                            grads_multi.append(grads_and_vars)
                        else:
                            # Get the testing output
                            probs_multi.append(tf.nn.softmax(pred[0]))

        # Combine the output probabilities
        if 'test' in networkNames: 
            self.prob['batch'] = tf.concat(probs_multi, axis=0)

        # Combine gradients, use weighting for augmentation
        grads = average_gradients(grads_multi)

        # Apply the gradients
        self.optimizer = optimizer.apply_gradients(grads, global_step=global_step)
        self.loss['batch'] = {
            name: tf.add_n(loss_multi[name]) / len(loss_multi[name])
                for name in networkNames
        }

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(max_to_keep=None)

        if self.opts.bool_display:
            # Create a list of subplots for each class, except the background (0)
            self.f1 = plt.figure()
            num_plots = self.opts.num_class - 1
            self.seg_plots = [[] for k in range(num_plots)]
            for k in range(num_plots):
                # Each class has a list of plots, one per channel
                for c in range(self.num_channels):
                    self.seg_plots[k].append(
                        self.f1.add_subplot(self.num_channels, num_plots, 
                            c * self.num_channels + k + 1)
                    )

        self.dataXX = np.zeros(xTr_size, dtype=np.float32)
        self.dataYY = np.zeros(yTr_size, dtype=np.int64)
        self.dataEE = np.zeros(yTr_size, dtype=np.int64)

        # Set up the session options
        config = tf.ConfigProto()
        config.allow_soft_placement = True # Soft device assignment
        config.gpu_options.allow_growth = True # Use only as much memory as we need

        self.sess = tf.Session(config=config)

        # Set up the session running options
        self.runOptions = tf.RunOptions(report_tensor_allocations_upon_oom = True) # Info for OOM

        # Training info
        self.train_iter = 0

    def limit_grads(self, grads, augFlags):
        # Takes in a list of grads and vars. Limits so that the step towards
        # augmentation is no larger than that towards real data.
        # (Theoretically, this would not move if we were perfectly overfit.)
        # 
        # This returns a final list of average grads

        # Take sub-averages for each group
        flags_range = range(len(augFlags))
        aug_grads = average_gradients([grads[i] for i in flags_range \
                if augFlags[i]])
        real_grads = average_gradients([grads[i] for i in flags_range \
                if not augFlags[i]])

        # Process each group and limit it
        final_grads = []
        for grad_and_vars in zip(aug_grads, real_grads):
            # Get the grads and an instance of the common variable
            aug_grad = grad_and_vars[0][0]
            real_grad = grad_and_vars[0][0]
            var = grad_and_vars[0][1]

            # Compute the limiting factor
            aug_norm = tf.sqrt(tf.reduce_sum(tf.square(aug_grad)))
            real_norm = tf.sqrt(tf.reduce_sum(tf.square(real_grad)))
            limiter = tf.minimum(real_norm / aug_norm, 1.0)

            # Compute the combined grad and add to the list
            final_grad = 0.5 * (aug_grad * limiter + real_grad)
            final_grads.append((final_grad, var))

        return final_grads

    def average_iou(self, pred, truth):
        #img_pred = np.argmax(pred, axis=3)
        #iou = 0.0
        #for i in range(1, pred.shape[-1]):
        #    intersection = np.sum((img_pred == i) & (truth == i))
        #    union = np.sum((img_pred == i) | (truth == i))
        #    iou += float(intersection) / float(union + 1) / (pred.shape[-1]-1)
        prob = np.squeeze(pred)
        gt = truth > 0.1
        intersection = np.sum(prob & gt)
        union = np.sum(prob | gt)
        iou = float(intersection) / float(max(union, 1))
        return iou

    def precision_recall(self, pred, truth):
        """
            Computes the detection precision and recall by taking connected
            components. A Optionally applies opening to the masks, to remove 
            small objects.

            An object is defined as a connected component in either mask. An
            object in 'pred' is a true positive if it overlaps with 'truth' by
            at least 50% of its volume. Similarly, a false negative is an
            object in 'truth' which overlaps with 'pred' by less than 50%.
        """
        import scipy.ndimage as nd

        # Optionally perform opening
        niter = max((self.opts.min_object_size - 1) / 2, 0)
        if niter > 0:
            pred = nd.morphology.binary_opening(pred, iterations=niter)
            truth = nd.morphology.binary_opening(truth, iterations=niter)

        # Get the connected components
        pred_labels, num_objects_pred = nd.label(pred)
        truth_labels, num_objects_truth = nd.label(truth)

        # Handle edge cases when one of the groups is empty
        if num_objects_pred == 0:
            if num_objects_truth == 0:
                return 1.0, 1.0
            else:
                return 0.0, 0.0
        elif num_objects_truth == 0:
            return 0.0, 0.0

        # Get the volume of each object in 'pred' and 'truth'
        pred_inds, volume_pred = np.unique(pred_labels, return_counts=True)
        truth_inds, volume_truth = np.unique(truth_labels, return_counts=True)

        # Remove the background object, label 0
        assert(pred_inds[0] == 0)
        assert(truth_inds[0] == 0)
        pred_inds = pred_inds[1:]
        volume_pred = volume_pred[1:]
        truth_inds = truth_inds[1:]
        volume_truth = volume_truth[1:]

        # Get the overlap percent for true positives, false negatives
        overlap_pos = np.array([np.sum((pred_labels == idx) & truth) \
            for idx in pred_inds]).astype(float) / volume_pred
        overlap_neg = np.array([np.sum((truth_labels == idx) & pred) \
            for idx in truth_inds]).astype(float) / volume_truth
        """
        num_true_pos = sum([np.any((pred_labels == idx) & truth) \
            for idx in range(1, num_objects_pred + 1)])
        num_false_neg = sum([not np.any((truth == idx) & pred) \
            for idx in range(1, num_objects_truth + 1)]) #XXX there was a bug here! Should be truth_labels, not 'truth'
        """

        # Apply the overlap threshold
        overlap_thresh = .5
        num_true_pos = np.sum(overlap_pos >= overlap_thresh)
        num_false_neg = np.sum(overlap_neg < overlap_thresh)

        precision = float(num_true_pos) / max(1, num_objects_pred)
        recall = float(num_true_pos) / max(1, num_false_neg + num_true_pos)

        return precision, recall
    
    def super_colormap(self, img, cmap):
        img -= np.min(img)
        imMax = np.max(img)
        if imMax > 0:
            img /= np.max(img)
        return_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.float32)
        red_chan = np.clip(-2 + 4.0*cmap, 0,1)
        green_chan = np.clip(2 - 4.0*np.abs(cmap - 0.5), 0,1)
        blue_chan = np.clip(2 - 4.0*cmap, 0,1)
        return_img[:,:,0] = 0.2*red_chan + 0.8*img
        return_img[:,:,1] = 0.2*green_chan + 0.8*img
        return_img[:,:,2] = 0.2*blue_chan + 0.8*img
        return return_img

    def super_colormap2(self, img, cmap, gt):
        img -= np.min(img)
        imMax = np.max(img)
        if imMax > 0:
            img /= np.max(img)
        if cmap.ndim == 3:
            cmap1 = cmap[:,:,1] #* (cmap[:,:,1] > cmap[:,:,2])
        else:
            cmap1 = cmap
        cmap_lt1 = cmap1 < 0.1
        cmap2 = np.zeros_like(cmap1)
        #cmap2 = cmap[:,:,2] * (cmap[:,:,2] > cmap[:,:,1])
        cmap_lt2 = cmap2 < 0.1
        gt1 = (gt == 1)
        gt1 = gt1 ^ scipy.ndimage.morphology.binary_erosion(gt1, iterations=1)
        gt1 = gt1 > 0.5
        gt2 = (gt == 2)
        gt2 = gt2 ^ scipy.ndimage.morphology.binary_erosion(gt2, iterations=1)
        gt2 = gt2 > 0.5
        return_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.float32)
        red_chan1 = np.clip(-2 + 4.0*cmap1, 0,1)
        red_chan2 = np.ones_like(cmap2)
        green_chan1 = np.clip(2 - 4.0*np.abs(cmap1 - 0.5), 0,1)
        green_chan2 = cmap2
        blue_chan1 = np.clip(2 - 4.0*cmap1, 0,1)
        blue_chan2 = 1.0 - cmap2
        red_chan1[cmap_lt1] = 0.0
        green_chan1[cmap_lt1] = 0.0
        blue_chan1[cmap_lt1] = 0.0
        red_chan2[cmap_lt2] = 0.0
        green_chan2[cmap_lt2] = 0.0
        blue_chan2[cmap_lt2] = 0.0
        red_chan = np.clip(red_chan1 + red_chan2, 0,1)
        green_chan = np.clip(green_chan1 + green_chan2, 0,1)
        blue_chan = np.clip(blue_chan1 + blue_chan2, 0,1)
        return_img[:,:,0] = 0.2*red_chan + 0.8*img
        return_img[:,:,1] = 0.2*green_chan + 0.8*img
        return_img[:,:,2] = 0.2*blue_chan + 0.8*img
        return_img[:,:,0][gt1] = 0.0
        return_img[:,:,1][gt1] = 1.0
        return_img[:,:,2][gt1] = 1.0
        return_img[:,:,0][gt2] = 1.0
        return_img[:,:,1][gt2] = 1.0
        return_img[:,:,2][gt2] = 0.0

        #print np.max(return_img), np.max(img)
        #print np.min(return_img), np.min(img)
        return return_img

    """
    Run plt.pause, but don't crash if it fails!
    """
    def safe_plot(self):

        # Quit if we have previously failed to plot
        if not self.have_display:
            return

        try:
            plt.pause(1e-5)
        except Exception:
            logging.debug(traceback.format_exc())
            self.have_display=False

    """
        Show all the class segmentations. If scaled=True, adjust multi-class
        segmentations so the probabilities are shown as if the only classes
        were the two top scorers.
    """
    def super_graph_seg(self, img, prob, truth, name='0', scaled=True):
        # Plot each class, except for the background (0)
        #FIXME generate a multi-color probability map, then just plot the
        # biggest slices here
        for k in range(1, self.opts.num_class):
            classProb = prob[:, :, :, k]
            if scaled is True:
                # Get the maximum probability for a class other than this one
                otherProb = np.delete(prob, k, axis=3)
                maxOtherProb = np.amax(otherProb, axis=3, keepdims=False)

                # Compute the scaled confidence score
                scaledProb = classProb / (classProb + maxOtherProb + 1e-10) 
            else:
                scaledProb = classProb

            # Graph each channel
            for c in range(self.num_channels):
                self.super_graph_seg_binary(
                    img, 
                    scaledProb, 
                    truth == k,
                    self.seg_plots[k - 1], 
                    name=name + " class %d" % k,
                    refresh=False
                )

        # Refresh the plots on the screen
        self.safe_plot()

    """
        Show a binary segmentation in the given plots. Takes in a list of
        plots, for each channel in img.
    """
    def super_graph_seg_binary(self, img, prob, truth, plotList, name='0', 
        refresh=True):
        # For volumetric segmentations, graph the largest axial slice
        img = img.squeeze()
        prob = prob.squeeze()
        truth = truth.squeeze()
        if truth.ndim == 3:
            truth_sum = np.sum(truth > 0, axis=0)
            truth_sum = np.sum(truth_sum, axis=0)
            slice_idx = np.random.choice(np.where(truth_sum == np.max(truth_sum))[0])
            img = img[:, :, slice_idx]
            prob = prob[:, :, slice_idx]
            truth = truth[:, :, slice_idx]

        for channel, plot in enumerate(plotList):
            # Break the image into channel components
            if len(plotList) > 1:
                chan = img[:, :, channel]
            else:
                chan = img.squeeze()

            # Plot each channel
            plot.cla()
            plot.imshow(self.super_colormap2(chan, prob, truth))
            plot.set_title(name)
        if refresh:
            self.safe_plot()

    def super_graph_seg2(self, img, img2, truth, save=True, name='0'):
        seg_sum = np.sum(truth, axis=0)
        seg_sum = np.sum(seg_sum, axis=0)
        slice_num = np.random.choice(np.where(seg_sum == np.max(seg_sum))[0])
        self.image_orig.cla()
        self.seg_pred.cla()
        self.seg_truth.cla()
        self.image_orig.imshow(img[:,:,slice_num], cmap='bone')
        self.seg_pred.imshow(img2[:,:,slice_num], cmap='bone')
        self.seg_truth.imshow(self.super_colormap(img[:,:,slice_num], truth[:,:,slice_num]))
        self.image_orig.set_title('Original')
        self.seg_pred.set_title('Directed Dream')
        self.seg_truth.set_title('Ground Truth')
        if self.opts.path_visualization and save:
            path_save = join(self.opts.path_visualization, 'DirectedDream')
            if not isdir(path_save):
                mkdir(path_save)
            self.f1.savefig(join(path_save, name + '.png'))
            m,n,s = img.shape
            img_orig = np.zeros((m*9+8,16*n+15))
            img_dream = np.zeros_like(img_orig)
            img_truth = np.zeros((m*9+8,16*n+15,3))
            for i in range(9):
                for j in range(16):
                    slice_num = int(float(16*i + j) * s / 144)
                    y_start = i*m + i
                    x_start = j*n + j
                    img_orig[y_start:(y_start+m),x_start:(x_start+n)] = img[:,:,slice_num]
                    img_dream[y_start:(y_start+m),x_start:(x_start+n)] = img2[:,:,slice_num]
                    img_truth[y_start:(y_start+m),x_start:(x_start+n),:] = self.super_colormap(img[:,:,slice_num], truth[:,:,slice_num])
            scipy.misc.imsave(join(path_save, name+'_orig.png'), img_orig)
            scipy.misc.imsave(join(path_save, name+'_dream.png'), img_dream)
            scipy.misc.imsave(join(path_save, name+'_truth.png'), img_truth)
            self.safe_plot()
        return 0

    def update_init(self):
        self.init = tf.global_variables_initializer()

    def super_print(self, statement, time=True):
        """
        This basically prints everything in statement.
        If time=True, add a time stamp to the statement.
        We'll print to stdout and path_log.
        """
        statement = str(statement)
        if time:
            statement = str(datetime.datetime.now()) + ' | ' + statement
        sys.stdout.write(statement + '\n')
        sys.stdout.flush()
        with open(self.opts.path_log, 'a') as f:
            f.write(statement + '\n')
        return 0

    def train_one_iter(self):
        """
        Basically trains one iteration.
        """

        # Adjust the amount of augmentation
        if self.opts.max_aug_iter > 1:
            amt =  min(float(self.train_iter) / self.opts.max_aug_iter, 1.)
        else:
            amt = 1.
      
        if self.opts.sampleProbs:
            # Draw a random batch of data indices, according using the pre-computed 
            # CDF
            uniform_draws = np.random.uniform(size=(self.opts.batch_size,))
            ind_list = np.searchsorted(self.sample_cdf, uniform_draws)
        else:
            ind_list = np.random.choice(range(len(self.train_filenames)), 
                self.opts.batch_size,
                replace=False)

        # Load this data
        volList = []
        segList = []
        extrasLists = {}
        for ind in ind_list:
            img_filename =  os.path.join(self.opts.path_train, self.train_filenames[ind])
            with h5py.File(img_filename, 'r') as hf:
                vol, seg, extras = self.h5Read(hf)

            # Save the data
            volList.append(vol)
            segList.append(seg)
            for key, val in extras.items():
                if key in extrasLists:
                    extrasLists[key].append(val)
                else: 
                    extrasLists[key] = [val]
        # Get the pre-processing parameters, optionally with data augmentation
        windowAmt = 10 # 100 for organs
        shapeList = [self.params['data_shape'] for vol in volList]
        xforms = [augment3d.get_xform(
            vol, 
            seg=seg,
            #rotMax=(45 * amt,) * 3,
            rotMax=(10 * amt,) * 3,
            #pReflect=(0.1 * amt,) * 3,
            pReflect=(0.05 * amt,) * 3,
            #shearMax=(1 + 0.1 * amt,),
            shearMax=(1 + 0.05 * amt,) * 3,
            transMax=(5 * amt,) * 3,
            #otherScale=0.05 * amt,
            otherScale=0.01 * amt,
            shape=shape,
            noiseLevel=self.opts.noiseLevel * amt,
            randomCrop=self.opts.cropping, # Defaults to 'uniform'
            windowMin=np.concatenate(
                (self.params['window_min'][np.newaxis] - windowAmt * amt, 
                self.params['window_min'][np.newaxis] + windowAmt * amt)
            ),
            windowMax=np.concatenate(
                (self.params['window_max'][np.newaxis] - windowAmt * amt, 
                self.params['window_max'][np.newaxis] + windowAmt * amt)
            ),
            #occludeProb=0.5 * amt,
            printFun=self.super_print
        ) if self.opts.augment else augment3d.get_xform(
            vol, 
            seg=seg,
            shape=shape,
            randomCrop=self.opts.cropping,
            windowMin=np.tile(self.params['window_min'], (2,1)),
            windowMax=np.tile(self.params['window_max'], (2,1))
        ) for vol, seg, shape in zip(volList, segList, shapeList)]

        # Apply the transforms to the image and segmentation
        volList, segList = augment3d.apply_xforms(
            xforms, 
            imList=volList, 
            labelsList=segList,
            oob_image=self.opts.oob_image_val,
            oob_label=self.opts.oob_label,
            api=self.opts.augment_api,
            device=self.opts.augment_gpu
        )

        # Apply the transforms to extras
        if self.per_object_code == perObjectVoronoiCode:

            if self.opts.oob_label > 0:
                raise ValueError("""Out-of-bounds not implemented for Voronoi 
                    diagrams""")

            # Augmentation
            voronoiList = augment3d.apply_xforms_labels(
                xforms,
                extrasLists['voronoi'],
                oob=-1,
                device=self.opts.augment_gpu
            )

            # Post-processing
            num_objects = np.zeros(self.opts.batch_size, np.int32)
            voronoi = np.zeros(self.dataYY.shape, voronoiList[0].dtype)
            for k in range(len(volList)):
                voronoi[k], num_objects[k] = reduceLabels(voronoiList[k], 
                    segList[k] >= 0) 
        elif self.per_object_code == perObjectFuzzyCode:

            if self.opts.oob_label > 0:
                raise ValueError("""Out-of-bounds not implemented for fuzzy 
                    diagrams""")

            # Get a set of default xforms for the weights, set the affines to
            # be the same as the old xforms
            fuzzyXforms = []
            for volIdx, xform in enumerate(xforms):
                weightVol = extrasLists['fuzzy_weights'][volIdx]
                outShape = xform['shape']
                if weightVol.ndim > 3:
                    outShape = outShape[:3] + (weightVol.shape[3],) # Adding channels
                fuzzyXform = augment3d.get_xform( # Getting a new xform
                    weightVol,
                    shape=outShape
                )
                fuzzyXform['affine'] = xform['affine'] # Copying affine
                fuzzyXforms.append(fuzzyXform)

            # Augment the weights
            fuzzyWeightsList = augment3d.apply_xforms_image(
                fuzzyXforms,
                extrasLists['fuzzy_weights'],
                oob=0,
                device=self.opts.augment_gpu
            )

            # Augment each channel of the labels separately
            #TODO refactoring could improve this
            fuzzyLabelsFlat = []
            flatXforms = []
            flatInds = [[] for x in extrasLists['fuzzy_labels']]
            for volNum, labelsVol in enumerate(extrasLists['fuzzy_labels']):
                for c in range(labelsVol.shape[3]):
                    flatInds[volNum].append(len(fuzzyLabelsFlat))
                    fuzzyLabelsFlat.append(labelsVol[:, :, :, c])
                    flatXforms.append(fuzzyXforms[volNum])
            fuzzyLabelsFlat = augment3d.apply_xforms_labels(
                flatXforms,
                fuzzyLabelsFlat,
                oob=-1,
                device=self.opts.augment_gpu
            )

            # Post-process and put all of this into the tensors
            num_objects = np.zeros(self.opts.batch_size, dtype=np.int32)
            fuzzyWeights = np.zeros(
                __tensor_shape__(self.fuzzy_weights['batch']),
                dtype=np.float32
            )
            fuzzyLabels = -np.ones(fuzzyWeights.shape, dtype=np.int32)
            for batchIdx, weightsVol in enumerate(fuzzyWeightsList):

                # Re-assemble the labels
                augLabels = -np.ones(weightsVol.shape, 
                    dtype=fuzzyLabelsFlat[0].dtype)
                for c in range(extrasLists['fuzzy_labels'][batchIdx].shape[3]):
                    augLabels[:, :, :, c] = fuzzyLabelsFlat[flatInds[batchIdx][c]]

                # Reduce the labels to remove absent objects
                reducedLabels, numObj = reduceLabels(augLabels, 
                    segList[batchIdx] >= 0)
                num_objects[batchIdx] = numObj
                numWriteObj = min(numObj, fuzzyNumVertices)
                fuzzyLabels[batchIdx, :, :, :, :reducedLabels.shape[-1]] = reducedLabels
                fuzzyWeights[batchIdx, :, :, :, :weightsVol.shape[-1]] = weightsVol
                # Note: unused objects have weight 0, label -1

        # Optional post-processing
        if self.opts.balanced:
            weight_map = np.zeros(self.dataYY.shape)
        for k in range(len(volList)):
            # Apply the mask
            seg = segList[k]
            if self.opts.masked:
                volList[k] = apply_mask(volList[k], seg != -1)

            # Compute the pixel weights
            if self.opts.balanced:
                weight_map[k] = get_weight_map(seg)

        # Assemble the final volumes and segmentations into a batch
        self.dataXX = np.stack(volList, axis=0)
        self.dataYY = np.stack(segList, axis=0)
        #XXX old code
        """
        for k, vol in enumerate(volList):
            self.dataXX[k] = vol
            self.dataYY[k] = segList[k]
        """

        # Possibly add a trailing singleton dimension
        if len(self.X['batch'].shape) > self.dataXX.ndim:
            self.dataXX = np.expand_dims(self.dataXX, self.dataXX.ndim)

        # Assemble the feed dict
        feed = {self.X['batch']: self.dataXX, 
                self.Y['batch'] : self.dataYY}
        if self.opts.balanced:
            feed[self.weight_map['batch']] = weight_map
        if self.have_per_object:
            feed[self.num_objects['batch']] = num_objects
        if self.per_object_code == perObjectVoronoiCode:
            feed[self.voronoi['batch']] = voronoi
        elif self.per_object_code == perObjectFuzzyCode:
            feed[self.fuzzy_weights['batch']] = fuzzyWeights
            feed[self.fuzzy_labels['batch']] = fuzzyLabels

        # Train
        _, loss_iter,seg_example = self.sess.run(
            (
                self.optimizer, 
                self.loss['batch']['train'], 
                self.segmentation_example
            ),
            feed_dict=feed,
            options=self.runOptions
        )

        """
        #XXX Check the voronoi
        for k in range(self.opts.batch_size):
            vor = voronoi[k]
            seg = self.dataYY[k]
            for j in range(num_objects[k]):
                assert(np.any((vor == j) & (seg >= 0)))
        """

        if self.opts.bool_display:
            self.super_graph_seg(self.dataXX[0], seg_example, self.dataYY[0], 
                    name="Training - " + self.train_filenames[ind_list[0]])

        # Count the iterations
        self.train_iter = self.train_iter + 1

        return loss_iter if not self.have_per_object \
            else loss_iter / np.sum(num_objects)

    def dream_one_iter(self, path_file, name='0'):
        raise ValueError("Dream is not implemented for 3D!!!!")
        dataXX = np.zeros((1,) + self.params['data_shape'])
        dataMASK = np.zeros((1,) + self.params['label_shape'] + (2,))
        try:
            with h5py.File(path_file) as hf:
                dataXX[0,:,:,:,:] = np.array(hf.get('data'))
                seg_array = np.array(hf.get('seg')).astype(bool)
                #seg_array = scipy.ndimage.morphology.binary_dilation(seg_array, iterations=10)
                seg_array = seg_array.astype(np.float32)
                dataMASK[0,:,:,:,0] = -1 * seg_array
                dataMASK[0,:,:,:,1] = seg_array
        except:
            print ('Failed: ' + path_file)
        feed = {self.X['single']:dataXX, self.mask:dataMASK}
        vol_seg = np.squeeze(dataMASK[0,:,:,:,1].copy())
        seg_sum = np.sum(vol_seg, axis=0)
        seg_sum = np.sum(seg_sum, axis=0)
        slice_num = np.random.choice(np.where(seg_sum == np.max(seg_sum))[0])
        dataXX_copy = dataXX.copy()
        img = dataXX[0,:,:,slice_num,0]
        grad = np.ones_like(dataXX)
        for i in range(1):
          dx = self.sess.run((self.saliency), feed_dict=feed)
          dx[0,:,:,:,0] = scipy.ndimage.filters.gaussian_filter(dx[0,:,:,:,0], sigma=2)
          dx /= np.std(dx)
          dx = np.clip(dx, -3.0, 3.0)
          dx += 3
          dx /= 6
          grad = 0.9 * grad + 0.1 * (dx ** 2)
          dataXX += 0.01 * dx / np.sqrt(grad + 0.000000001)
          dataXX = np.clip(dataXX, 0, 1.0)
          feed = {self.xTe:dataXX, self.mask:dataMASK}
        #dx = dataXX - dataXX_copy
        #rand = np.random.rand(mask.shape[0], mask.shape[1])
        if self.opts.bool_display:
            self.super_graph_seg(img, dx[0,:,:,slice_num,:], dataMASK[0,:,:,slice_num,1], save=True, name=name)
            #self.super_graph_seg2(dataXX_copy[0,:,:,:,0], dataXX[0,:,:,:,0], dataMASK[0,:,:,:,1], name=name)
        return dx[0,:,:,:,0]

    """
        Read an example from the h5 file
    """
    def h5Read(self, hf):

        # Read the data
        vol = np.array(hf.get('data'))
        seg = np.array(hf.get('seg'))
        extras = {}
        if self.per_object_code == perObjectVoronoiCode:
            extras['voronoi'] = np.array(hf.get('voronoi'))
        elif self.per_object_code == perObjectFuzzyCode:
            fuzzy_weights = __pad_channel__(np.array(hf.get('fuzzy_weights')))
            fuzzy_labels = __pad_channel__(np.array(hf.get('fuzzy_labels')))
            extras['fuzzy_weights'] = fuzzy_weights
            extras['fuzzy_labels'] = fuzzy_labels

            # Sanity checks
            assert(np.equal(np.array(fuzzy_weights.shape), 
                np.array(fuzzy_labels.shape)).all())
            assert(fuzzy_labels.shape[3] <= fuzzyNumVertices)

        # Collapse all classes to the range [-inf, num_class - 1]. Note that
        # negative labels are ignored. Labels above max_class are set to 0
        max_class = self.opts.num_class - 1
        seg[seg > max_class] = 0

        # Extend to 4D, for channels
        if len(vol.shape) == 3:
                vol = vol[:, :, :, np.newaxis]

        return vol, seg, extras

    def test_one_iter(self, path_file, name='0', save=False):
        """
        Does one forward pass and returns the segmentation. Optinally saves the
        input to self.path_visualization.
        INPUTS:
        - self: (object)
        - path_file: (str) path of the file to inference.
        """

        # Read from the h5
        with h5py.File(path_file) as hf:
            vol, seg, extras = self.h5Read(hf)

        # Optionally apply a mask
        if self.opts.masked:
            vol = apply_mask(vol, seg != -1)

        # Run testing, using tiles if necessary
        netType = 'batch'
        prob, seg_loss, vol_proc = inference.tile_inference(
            vol, [self.X[netType]], self.prob[netType], self.params, self.sess,
            labels=seg,
            labels_ph=self.Y[netType],
            weights_ph=self.weight_map[netType],
            loss_ph=self.loss[netType]['test'],
            voronoi=extras['voronoi'] if 'voronoi' in extras else None,
            voronoi_ph=self.voronoi[netType],
            num_objects_ph=self.num_objects[netType],
            fuzzy_weights=extras['fuzzy_weights'] if 'fuzzy_weights' in extras else None,
            fuzzy_weights_ph=self.fuzzy_weights[netType],
            fuzzy_labels=extras['fuzzy_labels'] if 'fuzzy_labels' in extras else None,
            fuzzy_labels_ph=self.fuzzy_labels[netType],
            device=self.opts.augment_gpu
        )

        # Compute the predictions by taking the max over all class probabilities
        pred = np.argmax(prob, axis=3)

        # Apply the mask to the output predictions
        pred[seg < 0] = -1

        # Compute the accuracies for each class
        num_class = self.opts.num_class
        iou = np.zeros(num_class - 1,)
        precision = np.zeros_like(iou)
        recall = np.zeros_like(iou)
        for k in range(1, num_class):
            idx = k - 1; # Don't store accuracy for class 0
            classPred = pred == k
            classTruth = seg == k
            iou[idx] = self.average_iou(classPred, classTruth)
            precision[idx], recall[idx] = self.precision_recall(classPred, 
                classTruth)

        # Optionally display the results 
        if self.opts.bool_display and self.have_display:
            self.super_graph_seg(vol_proc, prob, seg, name=name)

        # Optionally write the results
        if save:
            write_nii(
                os.path.join(self.opts.path_visualization, 
                    'vol_' + name + '.nii.gz'), 
                vol
            )
            write_nii(
                os.path.join(self.opts.path_visualization, 
                    'pred_' + name + '.nii.gz'), 
                pred.astype(np.float32)
            )
            write_nii(
                os.path.join(self.opts.path_visualization, 
                    'truth_' + name + '.nii.gz'), 
                seg.astype(np.float32)
            )

        return seg_loss, iou, precision, recall

    def test_all(self, path_X, one_per_dir=False):
        """
        Basically tests all the folders in path_X.
        INPUTS:
        - self: (object)
        - path_X: (str) file path to the data.
        """
        # Initializing variables.
        X_list = listdir(path_X)
        stats_shape = (self.opts.num_class - 1,)
        means = {
            'loss': 0.0,
            'iou': np.zeros(stats_shape),
            'dice': np.zeros(stats_shape),
            'precision': np.zeros(stats_shape),
            'recall': np.zeros(stats_shape)
        }
        statsKeys = [key for key, val in means.items()]
        sumSquares = {key: val for key, val in means.items()}
        num_test = 1

        # Doing the testing.
        for iter_data in range(len(X_list)):
            # Test
            path_file = join(path_X, X_list[iter_data])
            loss_iter, iou_iter, precision_iter, recall_iter = \
                self.test_one_iter(
                    path_file, 
                    name="Val - %d" % iter_data, 
                    save=True 
            )

            # Update stats
            iter_stats = {
                    'loss': loss_iter,
                    'iou': iou_iter,
                    'dice': iou2Dice(iou_iter),
                    'precision': precision_iter,
                    'recall': recall_iter
            }
            for key in means:
                means[key], sumSquares[key], new_num_test = __update_mean_var__(
                    means[key], sumSquares[key], num_test, iter_stats[key]
                )
            num_test = new_num_test

        # Return the mean and variance stats
        variances = {key: val / max(1, num_test - 1) for key, val in sumSquares.items()}
        return means, variances

    def save_model(self):
        """
            Save the model along with its parameters
        """
        # Save the model checkpoint and meta graph, for training
        self.saver.save(self.sess, self.opts.path_model)

        # Save the model parameters as a .pkl
        pkl_name = self.opts.path_model.replace('.ckpt', '.params.pkl')
        with open(pkl_name, 'wb') as f:
                pickle.dump(self.params, f, pickle.HIGHEST_PROTOCOL)

    def train_model(self):
        """
        Loads model and trains.
        """
        if not self.opts.path_train:
            return 0
        # Initializing
        start_time = time.time()
        loss_tr = 0.0
        if self.opts.bool_load:
            tf.reset_default_graph()
            self.saver.restore(self.sess, self.opts.path_model)
            self.super_print('Loaded weights from ' + self.opts.path_model)
        else:
            self.sess.run(self.init)
            self.super_print('Training from scratch')
        # Training
        self.super_print("Let's start the training!")
        loss_min = 1000000
        iou_max = 0.0
        niter = 0
        while True:
            loss_iter = self.train_one_iter()
            niter += 1
            current_time = time.time()
            print_time = (current_time - start_time) / 60
            self.super_print("Iter: %d Time: %f Training loss: %f" 
                % (niter, print_time, loss_iter))
            loss_tr += loss_iter / self.opts.epoch
            if ((niter - 1) % self.opts.epoch) == 0:
                if niter == 1:
                    loss_tr *= self.opts.epoch # Corrects for first-time loss
                statement = "\t"
                statement += "Iter: " + str(niter) + " "
                statement += "Time: " + str(print_time) + " "
                statement += "Loss_tr: " + str(loss_tr)
                if self.opts.validate:
                    self.super_print("====> Running validation...")
                    means, variances = self.test_all(self.opts.path_validation)
                    for key in means:
                        statement += " %s_val_mean: %s %s_val_var: %s " % (
                            key, str(means[key]), key, str(variances[key]))
                else:
                    self.super_print("====> Skipping validation...")

                # Get the saving loss criterion
                loss_crit = (means['loss'] if self.opts.criterion == 'val' 
                    else loss_tr)

                # Save the model with the lowest loss
                if loss_crit < loss_min and niter >= self.opts.epoch:
                    self.super_print("====> Saving model...")
                    loss_min = loss_crit
                    self.save_model()
                self.super_print(statement)
                loss_tr = 0

                # Optionally save a frozen checkpoint of the model
                if self.opts.save_frozen:
                    if not os.path.isdir(self.opts.path_frozen):
                        os.mkdir(self.opts.path_frozen)
                    frozen_name = "iter_%d.pb" % niter
                    frozen_path = os.path.join(self.opts.path_frozen, frozen_name)
                    freeze_graph(self.sess, frozen_path)
                    self.super_print("Froze model and saved to %s" % frozen_path)


    def do_dream(self):
        if not self.opts.path_inference:
            return 0
        # Initializing
        start_time = time.time()
        loss_te = 0.0
        self.saver.restore(self.sess, self.opts.path_model)
        for name_folder in listdir(self.opts.path_inference):
            path_imgs = join(self.opts.path_inference, name_folder)
            for name_img in listdir(path_imgs):
                if name_img[0] == '.':
                    continue
                if name_img[-3:] != '.h5':
                    continue
                path_file = join(path_imgs, name_img)
                mask = self.dream_one_iter(path_file, name=name_img[:-3])
                h5f = h5py.File(path_file, 'a')
                try:
                    h5f.create_dataset('seg_'+self.opts.name, data=mask)
                except:
                    del h5f['seg_'+self.opts.name]
                    h5f.create_dataset('seg_'+self.opts.name, data=mask)
                h5f.close()
        return 0
    
    def do_inference(self):
        """
        Loads model and does inference.
        """
        if not self.opts.path_inference:
            return 0
        # Initializing
        start_time = time.time()
        loss_te = 0.0
        self.saver.restore(self.sess, self.opts.path_model)
        for name_folder in listdir(self.opts.path_inference):
            path_imgs = join(self.opts.path_inference, name_folder)
            for name_img in listdir(path_imgs):
                if name_img[0] == '.':
                    continue
                if name_img[-3:] != '.h5':
                    continue
                path_file = join(path_imgs, name_img)
                #XXX
                raise ValueError("TODO: read the volume from the h5 file here")
                mask = self.inference_one_iter(vol)
                h5f = h5py.File(path_file, 'a')
                try:
                    h5f.create_dataset('seg_'+self.opts.name, data=mask)
                except:
                    del h5f['seg_'+self.opts.name]
                    h5f.create_dataset('seg_'+self.opts.name, data=mask)
                h5f.close()
