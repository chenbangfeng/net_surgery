-- -------------------------------
-- Created by Mahdyar Ravanbakhsh.
-- Date: 24/02/16 - 16.45
-- ------------------------------- 
-- Desc :
-- -------------------------------

require 'loadcaffe'
require("scripts.lua.common")

-- torch.setdefaulttensortype("torch.CudaTensor")

-- Load Model with CuDNN
--net = loadcaffe.load(proto_fcn_alexnet , model_fcn_alexnet ,'cudnn')
net = loadcaffe.load(proto_fcn_alexnet , model_fcn_alexnet )
--net = torch.load(th_model_fcn_alexnet)

torch.save(th_model_fcn_alexnet,net)

img = load_image('data/test.jpg', 300)
net:evaluate()
y = net:forward(img)
print(y:size())

