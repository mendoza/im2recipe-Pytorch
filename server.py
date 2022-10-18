from sklearn.preprocessing import normalize
from flask import Flask
from flask import request
import base64
from PIL import Image
from io import BytesIO
import numpy as np
import torchvision.transforms as transforms
from trijoint import im2recipe
import torch
import pickle
import os
import scripts.utils as utils
from flask_cors import CORS
import random
import lmdb

# data preparation
normalize_transf = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    # rescale the image keeping the original aspect ratio
    transforms.Resize(256),
    transforms.CenterCrop(224),  # we get only the center of that rescaled
    transforms.ToTensor(),
    normalize_transf])


def norm(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)


seed = 42

if not (torch.cuda.device_count()):
    device = torch.device(*('cpu', 0))
else:
    torch.cuda.manual_seed(seed)
    device = torch.device(*('cuda', 0))


model = im2recipe()
model.visionMLP = torch.nn.DataParallel(model.visionMLP)
model.to(device)

dataset = None
instr_vecs = None
names = None
idx2ind = None


def setup():
    global model
    global names
    global instr_vecs
    global idx2ind
    global dataset

    model_path = '/home/mendoza/Documents/im2recipe-pytorch/data/trained/ResNet50-new/model_e200_v-15.850.pth.tar'
    # load checkpoint
    print("=> loading checkpoint '{}'".format(model_path))
    if device.type == 'cpu':
        checkpoint = torch.load(
            model_path, encoding='latin1', map_location='cpu')
    else:
        checkpoint = torch.load(model_path, encoding='latin1')
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(model_path, checkpoint['epoch']))

    with open(os.path.join('results/rec_embeds.pkl'), 'rb') as f:
        instr_vecs = pickle.load(f)
    with open(os.path.join('results/rec_ids.pkl'), 'rb') as f:
        names = pickle.load(f)

    instr_vecs = normalize(instr_vecs)
    # load dataset
    print('Loading dataset.')
    dataset = utils.Layer.merge(
        [utils.Layer.L1, utils.Layer.L2, utils.Layer.INGRS], 'data/recipe1M')
    print("Done.")
    idx2ind = {}  # sample id to position in dataset
    for i in range(len(dataset)):
        idx2ind[dataset[i]['id']] = i

    ch = {'sushi': [], 'pie': [], 'pizza': [], 'lasagna': [], 'soup': [], 'burger': [],
          'pasta': [], 'salad': [], 'smoothie': [], 'cookie': []}

    for i, name in enumerate(names):
        title = dataset[idx2ind[name]]['title']
        for j, v in ch.items():
            if j in title.lower():
                ch[j].append(i)


def generate_visual_emb(img):
    global model
    visual_emb = model.visionMLP(img)
    visual_emb = visual_emb.view(visual_emb.size(0), -1)
    visual_emb = model.visual_embedding(visual_emb)
    visual_emb = norm(visual_emb)
    return visual_emb.data.cpu().numpy()


def rank(visual_emb):
    global model
    global names
    global instr_vecs
    global idx2ind
    q_vecs = np.full(
        shape=instr_vecs.shape,
        fill_value=visual_emb
    )
    d_vecs = instr_vecs

    # Ranker

    ids_sub = names
    sims = np.dot(q_vecs, d_vecs.T)  # for im2recipe
    # get a column of similarities
    sim_i2r = sims[0, :]
    ranked = np.argsort(sim_i2r)[::-1].tolist()
    # sort indices in descending order
    ranked_info = []
    for idx in ranked[:10]:
        ind_pred_i2r = ids_sub[idx]
        # find sample in database
        pred_i2r = dataset[idx2ind[ind_pred_i2r]]
        # retrieved ingrs
        ingrs = pred_i2r['ingredients']
        instrs = pred_i2r['instructions']
        title = pred_i2r['title']
        imgs = pred_i2r['images']
        ranked_info.append({
            'ingredients': ingrs,
            'instructions': instrs,
            'title': title,
            'imgs': imgs,
            'sim': str(round(sim_i2r[idx], 4))
        })
    return ranked_info


app = Flask(__name__)
CORS(app)


@app.route("/", methods=['POST'])
def test():
    request_json = request.get_json()
    base64_img = request_json['img']
    recipes = {}
    if base64_img != None:
        try:
            decoded = base64.b64decode(base64_img)
            img = Image.open(BytesIO(decoded))
            img = transform(img)
            img = img.view((1,)+img.shape)
            visual_emb = generate_visual_emb(img)
            recipes = rank(visual_emb)
        except Exception as e:
            print(e)

    return {'recipes': recipes}


def build_path(id):
    return os.path.join("./data/images/", "/".join(id[:4]), id)


if __name__ == "__main__":
    setup()
    mode = "BE"  # "BE"
    query = ""
    if mode == "CLI":
        with open(os.path.join("data", 'test_keys.pkl'), 'rb') as f:
            ids = pickle.load(f)
        sample = [x.encode('latin1') for x in ids]
        env = lmdb.open("data/test_lmdb", max_readers=1, readonly=True, lock=False,
                        readahead=False, meminit=False)
        ids = []
        for rec in sample:
            with env.begin(write=False) as txn:
                serialized_sample = txn.get(rec)
            sample = pickle.loads(serialized_sample, encoding='latin1')
            imgs = sample['imgs']
            curr_id = imgs[0].get("id")
            if curr_id != None:
                ids.append(curr_id)
        highest = -np.inf
        highest_id = -1
        for id in ids:
            query = build_path(id)
            if (os.path.isfile(query)):
                img = Image.open(query).convert('RGB')
                img = transform(img)
                img = img.view((1,)+img.shape)
                visual_emb = generate_visual_emb(img)
                recipes = rank(visual_emb)
                for recipe in recipes:
                    sim = recipe.get('sim')
                    if sim == None: continue
                    sim = float(sim)

                    if highest < sim:
                        highest = sim
                        highest_id = id
            else:
                print(f"File not found! id: {id} query: {query}")
        print(f"highest sim: {highest} id is {highest_id}")
    elif mode == "BE":
        app.run(debug=False)
    else:
        print("Only CLI and BE mode are implemented")
