-- --------------------------------
-- Created by Mahdyar Ravanbakhsh.
-- Date: 17/01/16 18.14
-- --------------------------------
-- Convert a caffe model to torch model
-- --------------------------------

require 'loadcaffe'
require("scripts.lua.common")

-- convert alex original
net_org = loadcaffe.load(proto_original_alex, model_original_alex,'cudnn')
print (net_org)
torch.save (th_model_original_alex, net_org)

img = load_image2batch('data/cat.jpg' , 10 , 227)
y_org = net_org:forward(img:cuda())

print (y_org:size())

-- convert alex-full-conv-992
net_conv = loadcaffe.load(proto_full_conv_992_alex, model_full_conv_992_alex, 'ccn2')
print (net_conv)
torch.save (th_model_full_conv_992_alex, net_conv)

img = load_image2batch('data/cat.jpg' , 32 , 227)
y_conv = net_conv:forward(img:cuda())

print (y_conv:size())

-- convert alex-full-conv-1000
net_conv = loadcaffe.load(proto_full_conv_1000_alex, model_full_conv_1000_alex,'cudnn')
print (net_conv)
torch.save (th_model_full_conv_1000_alex, net_conv)

img = load_image2batch('data/cat.jpg' , 10 , 227)
y_conv = net_conv:forward(img:cuda())

print (y_conv:size())
