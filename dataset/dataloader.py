import numpy as np
import os.path as osp
from tqdm import tqdm
from dataset.cityscapes import CityscapesDataset
from utils.file_op import load_config

cityscapes_config = load_config("../configs/cityscapes.yaml")


def create_dataset(dataset_name='Cityscapes', set='train'):
    support_dataset = ['Cityscapes', 'PASCAL_VOC', 'KITTI', 'BDD']
    support_set = ['train', 'val', 'test']
    assert support_dataset.__contains__(dataset_name)
    assert support_set.__contains__(set)

    if dataset_name == 'Cityscapes':
        dataset = CityscapesDataset(root=cityscapes_config['data_dir'],
                                    list_path=osp.join(cityscapes_config['list_dir'], set + '.txt'),
                                    crop_size=tuple(cityscapes_config['crop_size']),
                                    mean=tuple(cityscapes_config['mean']),
                                    std=tuple(cityscapes_config['std']), set=set,
                                    ignore_label=cityscapes_config['ignore_label'])

    return dataset


def class_weight(dataloader, num_classes):
    """
    calculate the class weight according to frequency
    """
    result = np.zeros(num_classes)
    print("calculating per-category weight")

    for sample in tqdm(dataloader):
        _, label = sample
        label = np.array(label)
        mask = (label >= 0) & (label < num_classes)
        result += np.bincount(label[mask].astype(np.uint8), minlength=num_classes)

    weights = []
    frequency = []
    num_pixels = np.sum(result)
    for cls_freq in result:
        frequency.append(cls_freq / num_pixels)
        weights.append(1 / (np.log(1.02 + (cls_freq / num_pixels))))

    return np.array(frequency), np.array(weights)

