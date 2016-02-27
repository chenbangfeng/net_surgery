
require("common_lua")
require("scripts.lua.common")

net = torch.load(th_model_original_alex)
print (net)

tmpfc6 = net:get(19)
convfc6= convertLinear2Conv1x1(tmpfc6,{6,6})
net.modules[19] = convfc6

tmpfc7 = net:get(22)
convfc7= convertLinear2Conv1x1(tmpfc7,{4096,4096})
net.modules[22] = convfc7

tmpfc8 = net:get(25)
convfc8= convertLinear2Conv1x1(tmpfc8,{4096,1000})
net.modules[25] = convfc8