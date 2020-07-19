import argparse
import sys

from utils.segmentation3d import segmentor

def main(args):
    """
    Main function to parse arguments.
    INPUTS:
    - args: (list of strings) command line arguments
    """
    # Reading command line arguments into parser.
    parser = argparse.ArgumentParser(description = 'Do CNN Segmentation.')

    # Paths: arguments for filepath to misc.
    parser.add_argument('--pTrain', dest='path_train', type=str, default=None)
    parser.add_argument('--pVal', dest='path_validation', type=str, default=None)
    parser.add_argument('--pTest', dest='path_test', type=str, default=None)
    parser.add_argument('--pInf', dest='path_inference', type=str, default=None)
    parser.add_argument('--pModel', dest='path_model', type=str, default=None)
    parser.add_argument('--pFrozen', dest='path_frozen', type=str, default=None)
    parser.add_argument('--pLog', dest='path_log', type=str, default=None)
    parser.add_argument('--pVis', dest='path_visualization', type=str, default=None)

    # Preprocessing parameters
    parser.add_argument('--window', dest='window', type=str, default=None)

    # Experiment Specific Parameters (i.e. architecture)
    parser.add_argument('--name', dest='name', type=str, default='noname')
    parser.add_argument('--net', dest='network', type=str, default='GoogLe')
    parser.add_argument('--nClass', dest='num_class', type=int, default=2)
    parser.add_argument('--cropSize', dest='crop_size', nargs='+', type=int, 
	default=None)
    parser.add_argument('--oob_label', dest='oob_label', type=int, default=-1)
    parser.add_argument('--oob_image_val', dest='oob_image_val', type=int, default=None)

    # Hyperparameters
    parser.add_argument('--l2', dest='l2', type=float, default=0.0000001)
    parser.add_argument('--l1', dest='l1', type=float, default=0.0)
    parser.add_argument('--bs', dest='batch_size', type=int, default=12)
    parser.add_argument('--loss', dest='loss_type', type=str, default='softmax')
    parser.add_argument('--per_object', dest='per_object', type=str, default='none')
    parser.add_argument('--iou_log_prob', dest='iou_log_prob', type=int, default=0)

    # GPU usage parameters
    parser.add_argument('--trainGPUs', dest='train_gpus', nargs='+', type=int, default=None)
    parser.add_argument('--augGPU', dest='augment_gpu', type=int, default=None)
        
    # Training parameters
    parser.add_argument('--lr', dest='lr', type=float, default=0.001)
    parser.add_argument('--dec', dest='lr_decay', type=float, default=1.0)
    parser.add_argument('--ep', dest='epoch', type=int, default=250)
    parser.add_argument('--maxAugIter', dest='max_aug_iter', type=int, default=-1)
    parser.add_argument('--criterion', dest='criterion', type=str, default='train')
    parser.add_argument('--cropping', dest='cropping', type=str, default='uniform')

    # Switches
    parser.add_argument('--bLo', dest='bool_load', type=int, default=0)
    parser.add_argument('--bDisplay', dest='bool_display', type=int, default=1)
    parser.add_argument('--bDream', dest='bool_dream', type=int, default=0)
    parser.add_argument('--bBatchNorm', dest='batch_norm', type=int, default=0)
    parser.add_argument('--bAutotune', dest='autotune', type=int, default=0)
    parser.add_argument('--bBalanced', dest='balanced', type=int, default=0)
    parser.add_argument('--bAugmentation', dest='augment', type=int, default=0)
    parser.add_argument('--bMasked', dest='masked', type=int, default=0)
    parser.add_argument('--bSampleProbs', dest='sampleProbs', type=int, default=0)
    parser.add_argument('--bValidate', dest='validate', type=int, default=1)
    parser.add_argument('--bFreeze', dest='save_frozen', type=int, default=0)
    parser.add_argument('--bBackgroundLoss', dest='background_loss', type=int, default=1)

    # Reporting parameters
    parser.add_argument('--minObjectSize', dest='min_object_size', type=int, 
	default=1)

    # Creating Object
    opts = parser.parse_args(args[1:])
    CNN_obj = segmentor(opts)
    CNN_obj.train_model() #Train/Validate the Model
    CNN_obj.test_model() #Test the Model.
    if opts.bool_dream:
        CNN_obj.do_dream()
    else:
        CNN_obj.do_inference() #Do inference on inference set.

    # We're done.
    return 0
    

if __name__ == '__main__':
    main(sys.argv)
