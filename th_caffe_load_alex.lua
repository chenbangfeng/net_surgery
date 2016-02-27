
require 'loadcaffe'
require("scripts.lua.common")

-- torch.setdefaulttensortype("torch.CudaTensor")

-- Load Model with CuDNN
net = loadcaffe.load(proto_original_alex, model_original_alex,'cudnn')

-- Forward an image to net
img = load_image2batch('data/test.jpg' , 10 , 227)
y = net:forward(img)

-- output result of net
y:size()
y[1]:size()
print (net)
out25 = net:get(25).output

-- backward net
go=torch.ones(100)
gi=net:backward(img,go)
print (gi:size())
image.display(gi)

disp = require 'display'
disp.image(img)
-- nodes = net:findModules('nn.Linear')
-- fc7 = nodes[#nodes-1]


-- Load Model with  with ccn2
require 'loadcaffe'
require 'ccn2'
require("scripts_lua.common_lua")
net = loadcaffe.load(proto_original_alex, model_original_alex,'ccn2')
img = load_image2batch('data/test.jpg' , 32 , 227)
y = net:forward(img:cuda())