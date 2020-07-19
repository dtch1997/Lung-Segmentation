#!/bin/bash

declare -r python='python -B -m pdb'

declare -r path_FCN=/home/dnr/Liver/CTH_seg/train_CNNsegmentation3d.py
declare -r path_code=/home/dnr/Liver
declare -r path_save=/data/pet-ct-vol/lung

declare -r name=Pet_CT_1mm_lung_half_ignore

declare -r path_model=$path_code/model
declare -r path_log=$path_code/logs
declare -r path_vis=$path_code/visualization
mkdir $path_model
mkdir $path_log
mkdir $path_vis

#declare -r network=chestCT
declare -r cropSize='80 80 112' # Lung aspect ratio
declare -r num_gpu=4
declare -r batch_size=12 # Divisible by # of GPUs
declare -r epoch=500 # The number of iterations per epoch
#declare -r lr=0.0005 # Stage -0,5
#declare -r lr=0.0001 # Stage 0
declare -r lr=0.00005 # Stage 0.5
#declare -r lr=0.00001 # Stage 1
#declare -r lr=0.000005 # Stage 1.5
#declare -r lr=0.0000025 # Stage 2
#declare -r lr=0.0000005 # Stage 3
declare -r net='uNet' # 'GoogLe' or 'uNet'
declare -r loadit=1
declare -r window='pet-ct-lung' # Sets preprocessing window. Options: 'liver', 'all', 'pet-ct-lung', 'pet-ct-all'
declare -r batch_norm=1 # Use batch norm if nonzero
declare -r autotune=0 # Allow Tensorflow to run benchmarks if nonzero
declare -r balanced=1 # If nonzero, weight the loss function to balance classes
declare -r min_object_size=5 # If > 1, objects smaller than this are not counted in detection
declare -r num_class=3 # Labels are clamped to the range (-inf, ..., num_class - 1]. Negative values are special 'ignore' labels
declare -r oob_label=-1 # Label for out-of-bounds voxels in data augmentation
declare -r oob_image_val=0 # Value for out-of-bounds voxels in the augmented image
declare -r display=1 # If nonzero, display training progress
declare -r augmentation=1 # If nonzero, augment data during training
declare -r maxAugIter=-1 # Takes this many iterations to reach peak augmentation. -1 to disable
declare -r loss='iou' # Options: 'iou' for IOU, 'softmax' for per-voxel cross-entropy
declare -r per_object='fuzzy' # If true, computes a separate loss for each object. Only applies to IOU
declare -r masked=0 # If true, set pixels with label -1 to zero
declare -r validate=1 # If true, run inference on the validation set and save the model with the highest accuracy. Otherwise, saves the model with the lowest training loss
declare -r criterion='train' # 'train' or 'val', decides which loss will save the model
declare -r cropping='valid' # 'uniform' (default), 'valid', 'none', etc.

##########
# TRAINING THE MODEL
##########
${python} $path_FCN --pTrain $path_save/training --pVal $path_save/validation --pModel $path_model/$name.ckpt --pLog $path_log/training.txt --pVis $path_vis --name $name --nGPU $num_gpu --bs $batch_size --ep $epoch --lr $lr --bLo $loadit --net $net --nClass $num_class --cropSize $cropSize --window $window --bBatchNorm $batch_norm --bAutotune $autotune --bBalanced $balanced --minObjectSize $min_object_size --oob_label $oob_label --oob_image_val $oob_image_val --bDisplay $display --bAugmentation $augmentation --maxAugIter $maxAugIter --loss $loss --bMasked $masked --bValidate $validate --criterion $criterion --per_object $per_object --cropping $cropping
