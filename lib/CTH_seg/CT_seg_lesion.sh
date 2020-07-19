#!/bin/bash

declare -r python='python -B'

declare -r path_FCN=/home/dnr/Liver/CTH_seg/train_CNNsegmentation3d.py
declare -r path_code=/home/dnr/Liver
declare -r path_data=/data/liver_lesion
declare -r path_frozen=/data/liver_lesion/frozen

#declare -r name=CT_Liver_Lesion_Unet_fuzzy
#declare -r name=CT_Liver_Lesion_Unet
declare -r name=CT_Liver_Lesion_Unet_ce

declare -r path_model=$path_code/model
declare -r path_log=$path_code/logs
declare -r path_vis=$path_code/visualization
mkdir $path_model
mkdir $path_log
mkdir $path_vis

#declare -r cropSize='144 144 144'
declare -r cropSize='120 120 120'
declare -r num_gpu=4
declare -r batch_size=6
#declare -r batch_size=3 # Usually use 6, but getting OOM with fuzzy loss
#declare -r epoch=250
declare -r epoch=50 # Previously 500, when validate=true
#declare -r lr=0.0001 # Stage 0
declare -r lr=0.00001 # Stage 1
#declare -r lr=0.0000025 # Stage 2
#declare -r lr=0.000001 # Stage 3
#declare -r lr=0.0000001 # Stage  4
declare -r net='uNet' # 'GoogLe' or 'uNet'
declare -r loadit=1
declare -r window='liver' # Sets preprocessing window. Options: 'liver'
declare -r batch_norm=1 # Use batch norm if nonzero
declare -r autotune=0 # Allow Tensorflow to run benchmarks if nonzero
declare -r balanced=1 # If nonzero, weight the loss function to balance classes
#declare -r min_object_size=5 # If > 1, objects smaller than this are not counted in detection
declare -r min_object_size=0 # If > 1, objects smaller than this are not counted in detection
declare -r num_class=2 # Labels are clamped to the range (-inf, ..., num_class - 1]. Negative values are special 'ignore' labels
declare -r oob_label=-1 # Label for out-of-bounds pixels in data augmentation
declare -r oob_image_val=0 # Like oob_label, but for voxels
declare -r augmentation=1 # If nonzero, augment data during training
declare -r maxAugIter=-1 # Takes this many iterations to reach peak augmentation. -1 to disable
declare -r loss='iou' # Options: 'iou' for IOU, 'softmax' for per-voxel cross-entropy
#declare -r per_object='fuzzy' # Changes the loss function. Options: 'voronoi', 'fuzzy', 'none' (default)
declare -r per_object='none' # Changes the loss function. Options: 'voronoi', 'fuzzy', 'none' (default)
declare -r display=1 # If nonzero, display training progress
declare -r masked=0 # If true, set pixels with label -1 to zero
declare -r validate=0 # If true, run inference on the validation set and save the model with the highest accuracy. Otherwise, saves the model with the lowest training loss
declare -r criterion='train' # 'train' or 'val', decides which loss will save the model
declare -r cropping='valid' # 'uniform' (default), 'valid', 'none', etc.
declare -r iou_log_prob=1 # For IOU loss, use squished log probabilities #XXX This is set to CE loss now!
declare -r freeze_model=1 # If true, freeze the weights and save at each epoch
declare -r background_loss=0 # If true, under IOU loss, include the background

##########
# TRAINING THE MODEL
##########
${python} $path_FCN --pTrain $path_data/training --pVal $path_data/validation --pModel $path_model/$name.ckpt --pLog $path_log/training.txt --pVis $path_vis --name $name --nGPU $num_gpu --bs $batch_size --ep $epoch --lr $lr --bLo $loadit --net $net --nClass $num_class --cropSize $cropSize --window $window --bBatchNorm $batch_norm --bAutotune $autotune --bBalanced $balanced --minObjectSize $min_object_size --oob_label $oob_label --oob_image_val $oob_image_val --bDisplay $display --bAugmentation $augmentation --maxAugIter $maxAugIter --loss $loss --bMasked $masked --bValidate $validate --criterion $criterion --iou_log_prob $iou_log_prob --bFreeze $freeze_model --pFrozen $path_frozen --bBackgroundLoss $background_loss
