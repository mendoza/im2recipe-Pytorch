import numpy as np
import matplotlib.pyplot as plt
import os
import json
from params import get_parser
from sklearn.preprocessing import normalize
import utils
import torchfile
from scipy.misc import imread, imresize
from shutil import copyfile
import pickle


def build_path_from_id(id):
    return os.path.join(*[x for x in id[:4]], id)


def load_layer(json_file):
    with open(json_file) as f_layer:
        return json.load(f_layer)


ch = {'sushi': [], 'pie': [], 'pizza': [], 'lasagna': [], 'soup': [], 'burger': [],
      'pasta': [], 'salad': [], 'smoothie': [], 'cookie': []}

parser = get_parser()
params = parser.parse_args()

# random.seed(params.seed)

DATA_ROOT = params.test_feats
IMPATH = '../data/images/'
partition = params.partition
with open(os.path.join('../cheese_cake.pkl'), 'rb') as f:
    cheese_img_vecs = pickle.load(f)
with open(os.path.join(params.test_feats, 'img_embeds.pkl'), 'rb') as f:
    im_vecs = pickle.load(f)
with open(os.path.join(params.test_feats, 'rec_embeds.pkl'), 'rb') as f:
    instr_vecs = pickle.load(f)
with open(os.path.join(params.test_feats, 'rec_ids.pkl'), 'rb') as f:
    names = pickle.load(f)

# im_vecs = normalize(im_vecs)
instr_vecs = normalize(instr_vecs)

# load dataset
print('Loading dataset.')
dataset = utils.Layer.merge(
    [utils.Layer.L1, utils.Layer.L2, utils.Layer.INGRS], params.dataset)
print("Done.")
idx2ind = {}  # sample id to position in dataset
for i in range(len(dataset)):
    idx2ind[dataset[i]['id']] = i

idx2ind_test = {}  # sample id to position in embedding matrix
for i in range(len(names)):
    idx2ind_test[names[i]] = i

for i, name in enumerate(names):
    title = dataset[idx2ind[name]]['title']
    for j, v in ch.items():
        if j in title.lower():
            ch[j].append(i)

q_vecs = np.full(
    shape=instr_vecs.shape,
    fill_value=cheese_img_vecs[0]
)
d_vecs = instr_vecs

# Ranker
N = 1000
idxs = range(N)
K = 8  # number of subplots
MAXLEN_INGRS = 20
SPN = 6  # text separation (y)
fsize = 20  # text size
max_n = 8  # max number of instructions & ingredients in the list
ref_y = 225

ids_sub = names
sims = np.dot(q_vecs, d_vecs.T)  # for im2recipe

# get a column of similarities
sim_i2r = sims[i, :]

# sort indices in descending order
ind_pred_i2r = ids_sub[np.argsort(sim_i2r)[::-1].tolist()[0]]

# find sample in database
pred_i2r = dataset[idx2ind[ind_pred_i2r]]
savedir = 'examples/'

# retrieved ingrs
ingrs = pred_i2r['ingredients']
instrs = pred_i2r['instructions']
with open(os.path.join(savedir, 'retr_title.txt'), 'w') as f:
    f.write("%s\n" % pred_i2r['title'])
with open(os.path.join(savedir, 'retr_ingrs.txt'), 'w') as f:
    for ingr in ingrs:
        f.write("%s\n" % ingr['text'])
with open(os.path.join(savedir, 'retr_instrs.txt'), 'w') as f:
    for instr in instrs:
        f.write("%s\n" % instr['text'])
