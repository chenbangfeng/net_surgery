-- -------------------------------
-- Created by Mahdyar Ravanbakhsh.
-- Date: 17/01/16 - 19.14
-- ------------------------------- 
-- Desc : global require libs and setups
-- -------------------------------

require 'cudnn'
require 'inn'
require 'image'
require 'torch'
require("scripts.lua.utils")
require("scripts.lua.create_models")
-- require("scripts_lua.utils")



-- common directories and files

proto_original_alex = 'caffe_proto/deploy.prototxt'
model_original_alex = 'caffe_models/bvlc_alexnet.caffemodel'
th_model_original_alex = 'torch_models/alexnet.net'

proto_full_conv_992_alex = 'caffe_proto/deploy_full_conv_992.prototxt'
model_full_conv_992_alex = 'caffe_models/bvlc_alexnet_full_conv_992.caffemodel'
th_model_full_conv_992_alex = 'torch_models/alexnet_full_conv_992.net'

proto_full_conv_1000_alex = 'caffe_proto/deploy_full_conv_1000.prototxt'
model_full_conv_1000_alex = 'caffe_models/bvlc_alexnet_full_conv_1000.caffemodel'
th_model_full_conv_1000_alex = 'torch_models/alexnet_full_conv_1000.net'

-- fcn-alex Trevor Darrell
proto_fcn_alexnet = 'caffe_proto/fcn32.prototxt'
model_fcn_alexnet = 'caffe_models/fcn32.caffemodel'
th_model_fcn_alexnet = 'torch_models/fcn32.net'

proto_fcn8s = 'caffe_proto/fcn-8s-pascal.prototxt'
model_fcn8s = 'caffe_models/fcn-8s-pascal.caffemodel'
th_model_fcn8s = 'torch_models/fcn-8s-pascal.net'

-- binary models
th_model_fc7_bin8 = 'torch_models/th_model_fc7_bin8.net'
th_model_fc6_bin8 = 'torch_models/th_model_fc6_bin8.net'