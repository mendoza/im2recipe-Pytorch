import pickle
from sqlite3 import Cursor
import lmdb
import math
import sys
from random import sample
sys.path.append("..")
from args import get_parser

parser = get_parser()
opts = parser.parse_args()

if opts.percentage == 1.0: exit(0)

def prepareSize(N):
    return math.ceil(N * opts.percentage)


keys = {'train' : [], 'val':[], 'test':[]}
print('loading keys...')
for k in keys.keys():
    with open('../data/{}_keys.pkl'.format(k),'rb') as f:
        keys[k] = pickle.load(f)

print('storing original sizes...')
original_sizes = {
    'train': len(keys['train']),
    'val': len(keys['val']),
    'test': len(keys['test'])
}

print('storing new sizes...')
new_sizes = {
    'train': prepareSize(original_sizes['train']),
    'val': prepareSize(original_sizes['val']),
    'test': prepareSize(original_sizes['test'])
}

print('calculating items to remove...')
keys_to_remove = {
    'train': sample(keys['train'], new_sizes['train']),
    'val': sample(keys['val'],new_sizes['val']), 
    'test': sample(keys['test'],new_sizes['test'])
}

env = {'train' : [], 'val':[], 'test':[]}
env['train'] = lmdb.open('../data/train_lmdb',map_size=int(1e11))
env['val']   = lmdb.open('../data/val_lmdb',map_size=int(1e11))
env['test']  = lmdb.open('../data/test_lmdb',map_size=int(1e11))

for partition in env.keys():
    print(f'Started with {env[partition].stat()["entries"]}')
    with env[partition].begin(write=True) as txn:
        with txn.cursor() as curs:
            for db_key in keys_to_remove[partition]:
                txn.delete(db_key.encode('latin1'))
    print(f'Finished with {env[partition].stat()["entries"]}')
    

env['train'].close()
env['val'].close()
env['test'].close()
print('saving keys...')
for k in keys_to_remove.keys():
    with open('../data/{}_keys.pkl'.format(k),'wb') as f:
        pickle.dump(list(set(keys[k]) - set(keys_to_remove[k]) ), f)