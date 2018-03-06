### Dataset for DLA

from collections import namedtuple
import json
from os.path import exists, join


Dataset = namedtuple('Dataset', ['model_hash', 'classes', 'mean', 'std',
                                 'eigval', 'eigvec', 'name'])

imagenet = Dataset(name='imagenet',
                   classes=1000,
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225],
                   eigval=[55.46, 4.794, 1.148],
                   eigvec=[[-0.5675, 0.7192, 0.4009],
                           [-0.5808, -0.0045, -0.8140],
                           [-0.5836, -0.6948, 0.4203]],
                   model_hash={'dla34': '6ba26179',
                               'dla46_c': '37675969',
                               'dla46x_c': '893b8958',
                               'dla60x_c': '5de0ad33',
                               'dla60': 'e2f4df06',
                               'dla60x': '3062c917',
                               'dla102': 'd94d9790',
                               'dla102x': 'ad62be81',
                               'dla102x2': '262837b6',
                               'dla169': '0914e092'})


def get_data(data_name):
    try:
        return globals()[data_name]
    except KeyError:
        return None


def load_dataset_info(data_dir, data_name='new_data'):
    info_path = join(data_dir, 'info.json')
    if not exists(info_path):
        return None
    info = json.load(open(info_path, 'r'))
    assert 'mean' in info and 'std' in info, \
        'mean and std are required for a dataset'
    data = Dataset(name=data_name, classes=0,
                   mean=None,
                   std=None,
                   eigval=None,
                   eigvec=None,
                   model_hash=dict())
    return data._replace(**info)
