import numpy as np
import tensorflow as tf
import os
import sys

from utils.segmentation3d import freeze_graph

"""
Loads the model as a meta graph, freezes all the weights and converts it to a 
.pb file. Usage:
    python freeze_graph.py [meta path] [pb_path] [checkpoint_path]

    Note that checkpoint_path is optional. If not provided, the program will
    search for the most recent checkpoint
"""

meta_path = sys.argv[1]
pb_path = sys.argv[2]

# Find the checkpoint file
if len(sys.argv) < 4:
    checkpoint_path = tf.train.latest_checkpoint(os.path.dirname(meta_path))
else:
    checkpoint_path = sys.argv[3]

# Restore the graph
saver = tf.train.import_meta_graph(meta_path, clear_devices=True)

with tf.Session() as sess:

    # Load the weights
    saver.restore(sess, checkpoint_path)

    # Freeze the graph
    freeze_graph(sess, pb_path)

