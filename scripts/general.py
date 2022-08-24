import pickle
from tabnanny import check
import lmdb
import sys
import torch
import numpy as np
from random import sample
sys.path.append("..")
from args import get_parser
from trijoint import im2recipe

parser = get_parser()
opts = parser.parse_args()


keys = {'train' : [], 'val':[], 'test':[]}
print('loading keys...')
for k in keys.keys():
    with open('../data/{}_keys.pkl'.format(k),'rb') as f:
        keys[k] = pickle.load(f)

env = {'train' : [], 'val':[], 'test':[]}
env['train'] = lmdb.open('../data/train_lmdb',map_size=int(1e11))
env['val']   = lmdb.open('../data/val_lmdb',map_size=int(1e11))
env['test']  = lmdb.open('../data/test_lmdb',map_size=int(1e11))

counter = 0
# for partition in env.keys():
partition = 'train'
with env[partition].begin() as txn:
    with txn.cursor() as curs:
        for key in keys[partition]:
            serialized_sample = txn.get(key.encode('latin1'))
            sample = pickle.loads(serialized_sample,encoding='latin1')
            if sample['classes'] - 1 == 0 :
                counter+= 1
                # with open('../data/classes1M.pkl','rb') as f:
                    # class_dict = pickle.load(f)
                    # id2class = pickle.load(f)
                    # print(id2class(key))
                    # print(class_dict[sample['classes']-1])

    
print(f"total class 0 {counter}")
env['train'].close()
env['val'].close()
env['test'].close()

# if not(torch.cuda.device_count()):
#     device = torch.device(*('cpu',0))
# else:
#     torch.cuda.manual_seed(opts.seed)
#     device = torch.device(*('cuda',0))
# model = im2recipe()
# model.visionMLP = torch.nn.DataParallel(model.visionMLP)
# model.to(device)

# vision_params = list(map(id, model.visionMLP.parameters()))
# base_params   = filter(lambda p: id(p) not in vision_params, model.parameters())
   
# # optimizer - with lr initialized accordingly
# optimizer = torch.optim.Adam([
#             {'params': base_params},
#             {'params': model.visionMLP.parameters(), 'lr': opts.lr*opts.freeVision }
#         ], lr=opts.lr*opts.freeRecipe)

# path = "../snapshots/model_e002_v--1.000.pth.tar"
# checkpoint = torch.load(path)
# model.load_state_dict(checkpoint['state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer'])
# input_var = checkpoint['input_var']
# target_var = checkpoint['target_var']
# output = checkpoint['original_output']
# criterion = checkpoint['criterion']
# print(output[3])
# print(target_var[2])
# print(output[2])
# print(target_var[1])

# cos_loss = criterion[0](output[0], output[1], target_var[0].float())
# img_loss = criterion[1](output[2], target_var[1])
# rec_loss = criterion[1](output[3], target_var[2])

# print(cos_loss)
# print(img_loss)
# print(rec_loss)
# loss =  opts.cos_weight * cos_loss +\
#         opts.cls_weight * img_loss +\
#         opts.cls_weight * rec_loss 

# optimizer.zero_grad()
# loss.backward()
# optimizer.step()
# l = [module for module in model.modules() if not isinstance(module, torch.nn.Sequential)]
# print(l)