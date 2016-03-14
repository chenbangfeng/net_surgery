# import caffe python path:
# import sys
# caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
# sys.path.insert(0, caffe_root + 'python')

###############################################################################
# Change standard alexnet to fully_conv model
# author : Mahdyar Ravanbaksh
###############################################################################

import caffe
import numpy as np
import py_surgery_utils

proto_original_alex = 'deploy.prototxt'
model_original_alex = 'caffe_models/bvlc_alexnet.caffemodel'

proto_full_conv_992_alex = 'deploy_full_conv_992.prototxt'
model_full_conv_992_alex = 'caffe_models/bvlc_alexnet_full_conv_992.caffemodel'

proto_full_conv_1000_alex = 'deploy_full_conv_1000.prototxt'
model_full_conv_1000_alex = \
    'caffe_models/bvlc_alexnet_full_conv_1000.caffemodel'

proto_fcn32 = 'caffe_proto/fcn32.prototxt'
model_fcn32 = 'caffe_models/fcn32.caffemodel'

proto_fcnalex_pascal = 'caffe_proto/fcn-alexnet-pascal.prototxt'
model_fcnalex_pascal = 'caffe_models/fcn-alexnet-pascal.caffemodel'

out_net_model = model_full_conv_1000_alex


# ============================================================================
# ============ surgery alex ==================================================

# -- First ---------------------------------------------------------
# Load the original net with fc
net = caffe.Net(proto_fcnalex_pascal, model_fcnalex_pascal, caffe.TEST)
# extract fcs
# for full_conv uncomment this
params = ['fc6', 'fc7', 'fc8']
# params = ['fc6', 'fc7']
# fc_params = {name: (weights, biases)}
fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data)
             for pr in params}

for fc in params:
    print '{} weights are {} dimensional and biases are {} dimensional' \
        .format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)

# -- Second ----------------------------------------------------------
# load fully conv prototxt
net_full_conv = caffe.Net(proto_full_conv_1000_alex, model_original_alex,
                          caffe.TEST)

# extract fc-conv s
# for full_conv uncomment this
params_full_conv = ['fc6-conv', 'fc7-conv', 'fc8-conv']
# params_full_conv = ['fc6-conv', 'fc7-conv']
# conv_params = {name: (weights, biases)}
conv_params = \
    {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data)
     for pr in params_full_conv}

for conv in params_full_conv:
    print '{} weights are {} dimensional and biases are {} dimensional' \
        .format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape)

# -- Third --------------------------------------------------------------
# copy weights and biases from original to fully-conv model
for pr, pr_conv in zip(params, params_full_conv):
    conv_params[pr_conv][0].flat = fc_params[pr][
        0].flat  # flat unrolls the arrays
    conv_params[pr_conv][1][...] = fc_params[pr][1]

# -- Finally ------------------------------------------------------------
# save fully-conv model
net_full_conv.save(out_net_model)


# ============================================================================
# ============ test image ====================================================

py_surgery_utils.forward_nets('cat.jpg', net, net_full_conv)

# ============================================================================
# ============ surgery evaluation ============================================

# fc7 distance
fc7 = net.blobs['fc7'].data
fc7_conv = net_full_conv.blobs['fc7-conv'].data
fc7_conv = np.reshape(fc7_conv, (10, 4096))
dist_fc7 = py_surgery_utils.compute_dist(fc7, fc7_conv)

np.savetxt('fc7_conv.out', fc7_conv, delimiter=',')
np.savetxt('fc7.out', fc7, delimiter=',')

# fc6 distance
fc6 = net.blobs['fc6'].data
fc6_conv = net_full_conv.blobs['fc6-conv'].data
fc6_conv = np.reshape(fc6_conv, (10, 4096))
dist_fc6 = py_surgery_utils.compute_dist(fc6, fc6_conv)

