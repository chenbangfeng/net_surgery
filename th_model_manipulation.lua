-- -------------------------------
-- Created by Mahdyar Ravanbakhsh.
-- Date: 18/01/16 - 19.01
-- ------------------------------- 
-- Desc : Replace Conv layers(fc7-bin,fc8-bin) binary for fc6-conv and fc7-conv outputs
-- -------------------------------

require("scripts.lua.common")
matio = require 'matio'

bin_size = 8
result_text_file = 'generate_conv_bin'..bin_size..'_models.txt'
img = load_image('data/cat.jpg' , 227)
net_conv = torch.load(th_model_full_conv_1000_alex)
res = "Original Model 'th_model_full_conv_1000_alex' \n"..model2text(net_conv).."\n"


-- remove fc8-conv and save new model th_model_fc7_bin8/16/24
print("removing softmax and save new model th_model_fc7_bin"..bin_size.."... \n")
net_conv:remove(22)

bin_mat = matio.load('output/itq_out/mat_fc7-'..bin_size..'.mat','project_mat')
convmodule = cudnn.SpatialConvolution(4096, bin_size, 1, 1, 1, 1, 0, 0, 1)
convmodule.weight:copy(bin_mat:transpose(1,2))
convmodule.name = 'fc8-bin'..bin_size
convmodule:cuda()
net_conv:add(convmodule)
torch.save(th_model_fc7_bin8, net_conv)
print(net_conv)
-- disable flips, dropouts and batch normalization
net_conv:evaluate()
y_conv = net_conv:forward(img:cuda())
print ("output size is:\n"..tostring(y_conv:size()))
res = res.."Remove fc8 new model 'th_model_fc7_bin"..bin_size.."' is : \n"
res = res..model2text(net_conv).."\noutput size is:\n"..tostring(y_conv:size()).."\n"


-- remove fc7-conv and save new model th_model_fc6_bin8/16/24
print("removing fc7 and save new model th_model_fc6_bin"..bin_size.."... \n")
net_conv:remove(22)
net_conv:remove(21)
net_conv:remove(20)
net_conv:remove(19)

bin_mat = matio.load('output/itq_out/mat_fc6-'..bin_size..'.mat','project_mat')
convmodule = cudnn.SpatialConvolution(4096, bin_size, 1, 1, 1, 1, 0, 0, 1)
convmodule.weight:copy(bin_mat:transpose(1,2))
convmodule.name = 'fc7-bin'..bin_size
convmodule:cuda()
net_conv:add(convmodule)
torch.save(th_model_fc6_bin8, net_conv)
print(net_conv)
-- disable flips, dropouts and batch normalization
net_conv:evaluate()
y_conv = net_conv:forward(img:cuda())
print ("output size is:\n"..tostring(y_conv:size()))
res = res.."Remove softmax new model 'th_model_fc6_bin"..bin_size.."' is : \n"
res= res..model2text(net_conv).."\noutput size is:\n"..tostring(y_conv:size()).."\n"

torch.save('output/'..result_text_file, res,'ascii')





-- get net summary
--net_conv_modules = net_conv.modules
--net_org = net_org.modules





