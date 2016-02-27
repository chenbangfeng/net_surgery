-- -------------------------------
-- Created by Mahdyar Ravanbakhsh.
-- Date: 25/02/16 - 18.21
-- ------------------------------- 
-- Desc :
-- -------------------------------


require("scripts.lua.common")
matio = require 'matio'
net = torch.load(th_model_fcn_alexnet)
bin_size=8

bin_mat = matio.load('output/itq_out/mat_fc7-'..bin_size..'.mat','project_mat')
convmodule = cudnn.SpatialConvolution(4096, bin_size, 1, 1, 1, 1, 0, 0, 1)
convmodule.weight:copy(bin_mat:transpose(1,2))
convmodule.name = 'fc8-bin'..bin_size
convmodule:cuda()
net:add(convmodule)
torch.save(th_model_fcn_alexnet, net)
print(net)

