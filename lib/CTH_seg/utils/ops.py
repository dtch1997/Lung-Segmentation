import numpy as np

# Compatibility with TF 2.0--this is TF 1 code
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except:
    import tensorflow as tf

import scipy.ndimage

from .layers import *

def get_saliency(pred, mask, x):
    dx = tf.gradients(pred, x, grad_ys=[mask])
    return dx

def get_L2_loss(reg_param, key="reg_variables"):
    """
    L2 Loss Layer. Usually will use "reg_variables" collection.
    INPUTS:
    - reg_param: (float) the lambda value for regularization.
    - key: (string) the key for the tf collection to get from.
    """
    L2_loss = 0.0
    for W in tf.get_collection(key):
        L2_loss += reg_param * tf.nn.l2_loss(W)
    return L2_loss

def get_L1_loss(reg_param, key="l1_variables"):
    """
    L1 Loss Layer. Usually will use "reg_variables" collection.
    INPUTS:
    - reg_param: (float) the lambda value for regularization.
    - key: (string) the key for the tf collection to get from.
    """
    L1_loss = 0.0
    for W in tf.get_collection(key):
        L1_loss += reg_param * tf.reduce_sum(tf.abs(W))
    return L1_loss

def get_seg_loss_weighted(logits, labels, num_class):
    logits = tf.reshape(logits, [-1, num_class])
    labels = tf.reshape(labels, [-1])
    return 0

def count_pixels(labels):
    counts = np.zeros_like(labels.astype(np.float32))
    for i in range(labels.shape[0]):
        count,_ = scipy.ndimage.measurements.label(labels[i,:,:,:].astype(int))
        count = count.astype(int)
        for j in range(1, np.max(count)+1):
            im_bw = (count == j).astype(np.float32)
            count[count == j] = 30.0 / np.cbrt(3.0 * np.sum(im_bw) / 4 / np.pi)
        counts[i,:,:,:] = count
    return counts    

def loss_det(reg, cls, labels):
    loss = 0.0
    for i in range(reg.shape[0]):
        labs_i,_ = scipy.ndimage.measurements.label(labels[i,:,:,:].astype(int))
        #list_labs = [(0,0,0,0) for i in range(np.max(labs_i)+1)]
        for j in range(1, np.max(labs_i)+1):
            im_bw = (labs_i == j)
            dia = np.cbrt(6.0 * np.sum(im_bw) / np.pi)
            r,c,s = scipy.ndimage.measurements.center_of_mass(im_bw)
            r_s = int(r/8)
            c_s = int(c/8)
            s_s = int(s/8)
            #list_labs[j] = (r,c,s,dia)
            #loss -= np.log(cls[i,r_s,c_s,s_s,0])
            d_r = (reg[i,r_s,c_s,s_s,0] - r/8 + r_s)
            d_c = (reg[i,r_s,c_s,s_s,1] - c/8 + c_s)
            d_s = (reg[i,r_s,c_s,s_s,2] - s/8 + s_s)
            d_d = (np.log(max(reg[i,r_s,c_s,s_s,3], 0.00000000001)) - np.log(dia/8))
            for d in [d_r,d_c,d_s,d_d]:
                if np.abs(d) < 1:
                    loss += 0.5 * d**2 / reg.shape[0]
                else:
                    loss += (np.abs(d) - 0.5) / reg.shape[0]

    #for i in range(cls.shape[0]):
    #    for j_r in range(cls.shape[1]):
    #        for j_c in range(cls.shape[2]):
    #            for j_s in range(cls.shape[3]):
    #                
    return np.float32(loss)

def loss_rpn(prob, labels):
    prob = prob[:,:,:,:,0]
    pred = (prob > 0.3)
    loss = 0
    for i in range(prob.shape[0]):
        labs_i,_ = scipy.ndimage.measurements.label(labels[i,:,:,:].astype(int))
        for j in range(1, np.max(labs_i)+1):
            im_bw = (labs_i == j)
            p_1 = np.mean(prob[i,:,:,:][im_bw])
            #if p_1 > 0.7:
            #    continue
            loss -= np.log(p_1 + 0.0000000001) / prob.shape[0]
        continue
        preds_i,_ = scipy.ndimage.measurements.label(pred[i,:,:,:].astype(int))
        preds_i = preds_i.astype(int)
        if np.max(preds_i) > 100:
            continue
        for j in range(1, np.max(preds_i)+1):
            im_bw = (preds_i==j)
            m,n,s = scipy.ndimage.measurements.center_of_mass(im_bw)
            #rad = np.cbrt(np.sum(im_bw))
            p_1 = np.mean(prob[i,:,:,:][im_bw])
            #if labels[i, int(m), int(n), int(s)] == 1:
            #    loss -= np.log(p_1)
            #else:
            if labels[i, int(m), int(n), int(s)] != 1:
                loss -= np.log(1 - p_1 + 0.000000001) / prob.shape[0]
    return np.float32(loss)

"""
    Computes the weighting map, to be used with get_softmax_loss(). This ensures
    that class 0 is balanced in weight with the positive classes. Negative
    classes are removed from the loss function.

    By default, volumes in which all voxels have the same label are assigned
    zero weight. To disable this, set ignore_uniform=False.
        
"""
def get_weight_map(labels, ignore_uniform=True):

    # Count the non-negative labels
    uniq, counts = np.unique(labels[labels >= 0], return_counts=True)

    # Return if there's only one category present
    num_classes = len(uniq)
    if num_classes < 2:
        if ignore_uniform:
            return np.zeros(labels.shape)
        else:
            return np.ones(labels.shape)

    # Assign equal weight to each class, normalized to a total weight of one 
    # per voxel
    num_valid = sum(counts)
    weights = (float(num_valid) / num_classes) / counts

    # Generate the weight map
    weight_map = np.zeros(labels.shape)
    for k in range(len(uniq)):
        weight_map[labels == uniq[k]] = weights[k]

    return weight_map

def __iou_class_loss__(class_prob, gt, weights=None, log_prob=False, ce_loss=None, valid=None):
    """
        Main IOU computation. Optional per-element weighting
    """

    # Loss versions
    balancedCe = True # Use the new version of CE loss

    if weights is not None:
        per_element_total *= weights
        per_element_intersection *= weights
    if ce_loss is not None:

        # Convert CE loss back to log probabilities
        ce_probs = -ce_loss;

        # Convert to double, to avoid underflow
        tf_dtype = tf.float64
        one = tf.constant(1.0, dtype=tf_dtype)
        ce_probs = tf.cast(ce_probs, dtype=tf_dtype)
        gt = tf.cast(gt, dtype=tf_dtype)
        valid = tf.cast(valid, dtype=tf_dtype)

        # Get the CE log probs
        sum_gt = tf.reduce_sum(gt)
        sum_ce_gt = tf.reduce_sum(ce_probs * gt)
        compliment = (one - gt) * valid
        sum_ce_compliment = tf.reduce_sum(ce_probs * compliment)
        if balancedCe:
            print('Using BALANCED CE loss')
            sum_compliment = tf.reduce_sum(compliment)
            """
            ce_intersection = 1.0 + sum_ce_gt / sum_gt # NaN on sum_gt == 0
            ce_union = 1.0 - sum_ce_compliment / sum_compliment # NaN on sum_compliment == 0
            """

            ce_intersection = sum_compliment * (sum_gt + sum_ce_gt)
            ce_union = sum_gt * (sum_compliment - sum_ce_compliment)

            # Check for degenerate losses, due to numerical stability issues
            degenerate = tf.logical_or(tf.less(sum_gt, one), 
                    tf.less(sum_compliment, one))

            # If degenerate, set to negative. This is ignored by the batch function
            ce_intersection = tf.where(degenerate, -one, ce_intersection)
            ce_union = tf.where(degenerate, one, ce_union)
        else:
            print('Using UNBALANCED CE loss')
            ce_intersection = sum_ce_gt + sum_gt
            ce_union = sum_gt - sum_ce_compliment
        gain = tf.cast(ce_intersection / ce_union, dtype=tf.float32)
    elif log_prob:
        print('Using LOG PROB loss')
        # Experimental!
        eps = 1e-10 # To avoid -inf in log. Make sure this is small, it bends the loss into a sigmoid
        sum_gt = tf.reduce_sum(gt)
        log_intersection = tf.reduce_sum(tf.math.log(class_prob + eps) * gt) + sum_gt
        # total = tf.reduce_sum(per_element_intersection) # Old version--this doesn't seem quite right
        # log_union = total - log_intersection # This was the old version--probably incorrect
        log_union = sum_gt - tf.reduce_sum(tf.math.log(1 - class_prob + eps) * (1 - gt))
        gain = log_intersection / log_union
    else:
        # Normal mode
        print('Using NORMAL IOU loss')
        per_element_intersection = class_prob * gt
        per_element_total = class_prob + gt
        total = tf.reduce_sum(per_element_total) 
        intersection = tf.reduce_sum(per_element_intersection)
        gain = intersection / (total - intersection)
    gain_an = tf.where(tf.is_nan(gain), 2.0, gain) # for NaN, set to negative
    return 1.0 - gain_an

def __accumulate_voronoi_iou_class_loss__(loop_idx, num_objects, loss,
    class_prob, gt, voronoi):
    """
        Loop body, for the Voronoi IOU loss.
    """

    # Apply the voronoi mask
    voronoi_mask = tf.cast(
        tf.equal(tf.cast(voronoi, dtype=tf.int32), 
            tf.cast(loop_idx, dtype=tf.int32)), 
        dtype=tf.float32
    )
    voronoi_class_prob = class_prob * voronoi_mask
    voronoi_gt = gt * tf.cast(voronoi_mask, dtype=gt.dtype)

    # Get the new loss
    new_loss = loss + __iou_class_loss__(voronoi_class_prob, voronoi_gt)

    return loop_idx + 1, num_objects, new_loss, class_prob, gt, voronoi

def __accumulate_fuzzy_iou_class_loss__(loop_idx, num_objects, loss, class_prob,
        gt, fuzzy_weights, fuzzy_labels):
    """
        Loop body for fuzzy IOU loss.
    """

    # Get the weights belonging to this object, all others set to 0
    obj_mask = tf.cast(tf.equal(fuzzy_labels, loop_idx), dtype=tf.float32)
    obj_weights = fuzzy_weights * obj_mask

    # Collapse the channels by adding together all the weights. For each voxel,
    # at most one channel is nonzero
    collapse_weights = tf.reduce_sum(obj_weights, axis=1)

    # Get the new loss
    new_loss = loss + __iou_class_loss__(class_prob, gt, 
        weights=collapse_weights)

    return (loop_idx + 1, num_objects, new_loss, class_prob, gt, fuzzy_weights,
        fuzzy_labels)

def __loop_cond__(*args):
    """
        Loop condition, see __accumulate_voronoi_iou_class_loss__.
    """
    idx = args[0]
    max_iter = args[1]
    return tf.squeeze(idx < max_iter)

def get_iou_loss_batch(logits, labels, num_class, **kwargs):
    """
        Like get_iou_loss, but takes the loss over the batch index separately.
    """
    #TODO refactor with pattern: arg[k] if arg is not None else None.
    # Just need to remove this ridiculous tuple(logits[k]) thing.
    batch_size = int(labels.shape[0])
    losses = []
    for k in range(batch_size):
        batch_logits = tuple([logit[k] for logit in logits])
        batch_kwargs = {key: val[k] for key, val in kwargs.items() \
            if val is not None}
        losses.append(get_iou_loss(batch_logits, labels[k], num_class, 
            **batch_kwargs))

    return sum(losses) / float(len(losses))
    
def get_iou_loss(logits, labels, num_class, voronoi=None, 
    num_objects=None, fuzzy_weights=None, fuzzy_labels=None, 
    log_prob=False, background_loss=False):
    """
    Takes 1 - IOU, the intersection over union. See get_softmax_loss for the
    parameters.
    """

    # Verify inputs
    if fuzzy_weights is not None and fuzzy_labels is None or \
            fuzzy_labels is not None and fuzzy_weights is None:
        raise ValueError('Cannot have fuzzy_weights without fuzzy_labels')
    if fuzzy_weights is not None and voronoi is not None:
        raise ValueError('Cannot have both fuzzy_weights and voronoi')

    # Set up special loss variants
    if voronoi is not None:
        loopBody = __accumulate_voronoi_iou_class_loss__ 
    elif fuzzy_weights is not None:
        loopBody = __accumulate_fuzzy_iou_class_loss__
    else:
        loopBody = None

    # Put the input in a list for compatiblity
    try:
        [logit for logit in logits]
    except TypeError:
        logits = [logits] 
    # Flatten the input arrays
    labels = tf.reshape(labels, [-1])
    logits = [tf.reshape(logit, [-1, num_class]) for logit in logits]
    if voronoi is not None:
        voronoi = tf.reshape(voronoi, [-1])
    if fuzzy_weights is not None:
        fuzzy_weights = tf.reshape(fuzzy_weights, [-1, 4])
    if fuzzy_labels is not None:
        fuzzy_labels = tf.reshape(fuzzy_labels, [-1, 4])

    # Remove the invalid (masked out) voxels
    valid = tf.cast(labels >= 0, dtype=tf.float32)

    # Compute the loss for each probability map
    losses = []
    for logit in logits:
        # Convert the logits to softmax probabilities, mask out invalid
        probs = tf.nn.softmax(logit)

        ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=tf.reshape(logit, [-1, num_class]),
            labels=tf.maximum(labels, 0)
        ) * valid if log_prob else None

        # Compute the loss for each POSITIVE class label. Class 0 is implicit
        # EXCEPT for log prob mode
        #start_class = 0 if log_prob else 1
        start_class = 0 if background_loss else 1
        for class_idx in range(start_class, num_class):
            # Isolate the ppobabilities for this class
            class_prob = probs[:, class_idx]

            # Apply the mask
            class_prob *= valid

            gt = tf.cast(tf.equal(labels, class_idx), dtype=tf.float32)
            if class_idx < 1 or loopBody is None:
                #XXX This gives the CE
                class_loss = __iou_class_loss__(class_prob, gt, ce_loss=ce_loss, 
                        valid=valid)
            else: # Applies only to positive class labels, uses the voronoi
                # Computes the loss for each voronoi cell, for this class
                object_iter = tf.constant(0, dtype=tf.int32,
                    name="object_iter%d" % class_idx)
                loss_acc = tf.constant(0.0, dtype=tf.float32,
                    name="class_loss%d" % class_idx)

                # Set up the arguments
                if voronoi is not None:
                    loopArgs = [object_iter, num_objects, loss_acc,
                        class_prob, gt, voronoi]
                elif fuzzy_weights is not None:
                    loopArgs = [object_iter, num_objects, loss_acc, 
                        class_prob, gt, fuzzy_weights, fuzzy_labels]

                # Make the loop
                class_loss = tf.while_loop(
                    __loop_cond__,
                    loopBody, 
                    loopArgs,
                    back_prop=True
                )[loopArgs.index(loss_acc)]
            losses.append(class_loss)

    # Average all the losses, ignoring negatives
    negatives = [tf.less(loss, 0.0) for loss in losses]
    losses_ignore = [tf.where(negative, 0.0, loss) for loss, negative in zip(losses, negatives)]
    positives = [tf.cast(tf.logical_not(negative), dtype=tf.float32) for negative in negatives]
    sum_positives = sum(positives)
    avg_loss = sum(losses_ignore) / sum_positives

    # Return the final loss, handling division by zero in case NO class is represented
    return tf.where(tf.less(sum_positives, 1.0), 0.0, avg_loss)

def verify_iou_loss(pred, labels, num_class):
    """
    Numpy IOU loss implementation, for debugging
    """

    labels = np.reshape(labels, -1)
    valid = tf.cast(labels >= 0, dtype=tf.float32)
    pred = np.reshape(pred, [-1, num_class])
    losses = np.zeros((num_class,))
    for class_idx in range(num_class):
        prob = pred[:, class_idx] * valid
        gt = (labels == class_idx).astype(np.float32)
        total = np.sum(prob) + np.sum(gt)
        intersection = np.sum(prob * gt)
        losses[class_idx] = 1.0 - (intersection / (total - intersection))

    return np.mean(losses)

def get_softmax_loss(logits, labels, num_class, weight_map=None):#spheres, num_class):
    """
    Takes the softmax loss of the provided segementation maps (logits), then 
    averages them.

    Logits are assumed to be multiple probability maps, and the final loss is
    the average of all of their individual losses. In most cases, logits is
    just a single map.
    """

    # Put the input in a list for compatiblity
    try:
        [logit for logit in logits]
    except TypeError:
        logits = [logits]

    # Compute the loss, substituting 0 for negative labels
    labels = tf.reshape(labels, [-1])#tf.cast(tf.reshape(labels, [-1]), tf.float32)
    losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=tf.reshape(logit, [-1, num_class]),
        labels=tf.maximum(labels, 0)
    ) for logit in logits]

    # Average all the losses
    loss = sum(losses) / float(len(losses))

    # Ignore the dummy values, and multiply the loss accordingly
    valid = tf.cast(labels >= 0, dtype=tf.float32)
    loss *= valid 

    # Apply the weighting map
    if weight_map is not None:  
        loss *= tf.reshape(weight_map, [-1])

    # Average the loss, and scale up to account for ignored voxels      
    numel = np.prod(loss.shape.as_list())
    loss_scale = numel / tf.cast(
        tf.maximum(tf.count_nonzero(valid), 1), tf.float32
    )
    return tf.reduce_mean(loss) * loss_scale

def get_ce_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

def get_accuracy(logits, labels):
    """
    Calculates accuracy of predictions.  Softmax based on largest.
    INPUTS:
    - logits: (tensor.2d) logit probability values.
    - labels: (array of ints) basically, label \in {0,...,L-1}
    """
    pred_labels = tf.argmax(logits,1)
    correct_pred = tf.equal(pred_labels, labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy

def get_optimizer(lr, decay, epoch_every):
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = float(lr)
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               epoch_every, decay, staircase=True)
    optimizer = tf.train.RMSPropOptimizer(learning_rate)
    return optimizer, global_step
