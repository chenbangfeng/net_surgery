<a name="alex.intro"></a>
## Alexnet Surgery ##

`Alexnet Surgery` is a set of tools to import caffe model and manipulate alex
 net.
 scripts in python (for caffe) and lua (for torch). </br>
 

<a name="alex.dirstructure"></a>
## Directories ##

* *caffe_models* : 
binary caffe alexnet models.

* *caffe_proto* : 
deploy prototxt files for caffe alexnet.

* *torch_models* : 
binary torch alexnet models.

* *torch_proto* : 
torch text different alexnet structures.

* *scripts* : 
`lua` shared script folder. for import in new script and use functions :
```lua
require("scripts.lua.common")
require("scripts.lua.utils")
require("scripts.lua.create_models")
```
`common` includes common directories addresses, required libraries and also 
`utils` and `create_models` functions. </br>
`utils` includes frequently used functions such as load image, converters, etc.
</br>
`create_models` includes functions to create torch models.

* *data* : 
test sample data and images (input data).

* *output* : 
outputs and reports.

<a name="alex.models"></a>
## Models ##

to download pretrained models from dropbox run `getmodels.sh` script from 
folder `/torch/load_caffe/alex` : </br>
```
~/torch/load_caffe/alex $ bash ./getmodels.sh
```
<a name="alex.luascripts"></a>
## Lua Scripts ##

* *th_convert_fc2conv_alex* : 
Change `fc6`, `fc7` and `fc8` alexnet original to fully_conv1x1 modules

* *th_convert_model_caffe2torch* : 
Convert caffe models (standard, full_conv_fc8_992, full_conv_fc8_1000) to torch 
models and save.

* *th_model_manipulation* :
Manipulating alexnet_fullconv remove fc8-conv and fc7-conv, replace with binary layers(8,16,24bits).
Generating and save new models `th_model_fc7_bin8` , `th_model_fc6_bin8`.

* *th_models_evaluation* :
Evaluating fc6, fc7, fc8 outputs for standard alex model and full_conv version.

<a name="alex.pyscript"></a>
## Python Scripts ##

* *py_surgery_utils* : 
common functions and utilities.

* *py_alex_surgery* : 
Change standard alexnet to fully_conv model in caffe.
