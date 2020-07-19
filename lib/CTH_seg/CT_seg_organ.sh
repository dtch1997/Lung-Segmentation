#!/bin/bash

declare -r python='python -B -m pdb'

declare -r script=train_CNNsegmentation3d.py

declare -r path_data=/data/organ
declare -r path_frozen=$path_data/frozen

declare -r name=CT_Organ_3mm_with_gb

declare -r path_model=$path_data/model

declare -r path_log=$path_model/log_$name.txt

declare -r path_vis=$path_data/visualization
mkdir $path_model
mkdir $path_log
mkdir $path_vis

#declare -r network=chestCT
declare -r cropSize='120 120 160' # for 3mm
#declare -r train_gpus='0 1 2 3 4' # The GPUs to use for training
#declare -r augment_gpu=5 # The GPU to use for data augmentation. If empty, uses the CPU
#declare -r batch_size=5 # Divisible by # of train GPUs
declare -r train_gpus='0 1 2' # The GPUs to use for training
declare -r augment_gpu=3 # The GPU to use for data augmentation. If empty, uses the CPU
declare -r batch_size=3 # Divisible by # of train GPUs
#declare -r epoch=500
declare -r epoch=50
declare -r lr=0.0001 # Stage 0
#declare -r lr=0.00005 # Stage 0.5
#declare -r lr=0.00001 # Stage 1
#declare -r lr=0.000005 # Stage 1.5
#declare -r lr=0.0000025 # Stage 2
#declare -r lr=0.0000005 # Stage 3
declare -r net='uNet' # 'GoogLe' or 'uNet'
declare -r loadit=0
declare -r window='all' # Sets preprocessing window. Options: 'liver', 'all'
declare -r batch_norm=1 # Use batch norm if nonzero
declare -r autotune=0 # Allow Tensorflow to run benchmarks if nonzero
declare -r balanced=1 # If nonzero, weight the loss function to balance classes
declare -r min_object_size=5 # If > 1, objects smaller than this are not counted in detection
declare -r num_class=8 # Labels are clamped to the range (-inf, ..., num_class - 1]. Negative values are special 'ignore' labels
declare -r oob_label=-1 # Label for out-of-bounds voxels in data augmentation
declare -r oob_image_val=0 # Value for out-of-bounds voxels in the augmented image
declare -r display=1 # If nonzero, display training progress
declare -r augmentation=0 # If nonzero, augment data during training
declare -r maxAugIter=-1 # Takes this many iterations to reach peak augmentation. -1 to disable
declare -r loss='iou' # Options: 'iou' for IOU, 'softmax' for per-voxel cross-entropy
declare -r masked=0 # If true, set pixels with label -1 to zero
declare -r validate=0 # If true, run inference on the validation set and save the model with the highest accuracy. Otherwise, saves the model with the lowest training loss
declare -r criterion='train' # 'train' or 'val', decides which loss will save the model
declare -r iou_log_prob=0 # For IOU loss, use squished log probabilities #XXX This is set to CE loss now!
declare -r freeze_model=1 # If true, freeze the weights and save at each epoch
declare -r background_loss=1 # If true, under IOU loss, include the background

##########
# TRAINING THE MODEL
##########
${python} $script --pTrain $path_data/training --pVal $path_data/validation --pModel $path_model/$name.ckpt --pLog $path_log/training.txt --pVis $path_vis --name $name --trainGPUs $train_gpus --augGPU $augment_gpu --bs $batch_size --ep $epoch --lr $lr --bLo $loadit --net $net --nClass $num_class --cropSize $cropSize --window $window --bBatchNorm $batch_norm --bAutotune $autotune --bBalanced $balanced --minObjectSize $min_object_size --oob_label $oob_label --oob_image_val $oob_image_val --bDisplay $display --bAugmentation $augmentation --maxAugIter $maxAugIter --loss $loss --bMasked $masked --bValidate $validate --criterion $criterion --iou_log_prob $iou_log_prob --bFreeze $freeze_model --pFrozen $path_frozen --bBackgroundLoss $background_loss


