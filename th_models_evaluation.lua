-- -------------------------------
-- Created by Mahdyar Ravanbakhsh.
-- Date: 18/01/16 - 20.12
-- ------------------------------- 
-- Desc : Evaluating standard alex model and full_conv version
-- -------------------------------

require 'loadcaffe'
require("scripts.lua.common")
result_text_file = 'eval_fcs_alex_standard_fullconv.txt'

-- load torch models
print ('Loading models...')
net_org = torch.load(th_model_original_alex)
net_conv = torch.load(th_model_full_conv_1000_alex)
-- disable flips, dropouts and batch normalization
net_org:evaluate()
net_conv:evaluate()

print("Original Alex net : \n", net_org)
print("Fully_conv Alex net : \n", net_conv)
res = "##### Original Alex net #####\n"
res = res..model2text(net_org).."\n"
res = res.."##### Fully_conv Alex net #####\n"
res = res..model2text(net_conv).."\n"

-- Forward a sample image for evaluation
print ('Start evaluation...')
img = load_image2batch('data/cat.jpg' , 10 , 227)
y_conv = net_conv:forward(img:cuda())
y_org = net_org:forward(img:cuda())

-- evaluate fc6, fc7 to fc6_conv, fc7_conv
-- fc6_conv : (16) --- fc6 : (17)
-- fc7_conv : (19) --- fc7 : (20)
-- fc7_conv : (22) --- fc7 : (23)
print ('Evaluation results on fc6, fc7, fc8 for alexnet standard and full_conv')
eval_fc6 = compute_dist(net_conv:get(16).output:reshape(10,4096),net_org:get(17).output)
print('-fc6: Maximum distance : ', eval_fc6:max(), ' mean distances : ', eval_fc6:mean())
eval_fc7 = compute_dist(net_conv:get(19).output:reshape(10,4096),net_org:get(20).output)
print('-fc7: Maximum distance : ', eval_fc7:max(), ' mean distances : ', eval_fc7:mean())
eval_fc8 = compute_dist(net_conv:get(22).output:reshape(10,1000),net_org:get(23).output)
print('-fc8: Maximum distance : ', eval_fc8:max(), ' mean distances : ', eval_fc8:mean())

print ('Saving results to: output/'..result_text_file)
res = res..'Evaluating standard alex model and full_conv version\n'
res = res..'Evaluation results on fc6, fc7, fc8\n'
res = res..'-fc6: Maximum distance : '..eval_fc6:max()..'\tmean distances : '..eval_fc6:mean()..'\n'
res = res..'-fc7: Maximum distance : '..eval_fc7:max()..'\tmean distances : '..eval_fc7:mean()..'\n'
res = res..'-fc8: Maximum distance : '..eval_fc8:max()..'\tmean distances : '..eval_fc8:mean()..'\n'
torch.save('output/'..result_text_file, res,'ascii')