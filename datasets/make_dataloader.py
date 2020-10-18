import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .naic import NAIC
from .veri import VeRi
from .contest import CONTEST
from .contest2019 import CONTEST2019
from .bases import ImageDataset
from .preprocessing import RandomErasing, RandomPatch, RandomGaussianBlur
from .sampler import RandomIdentitySampler

__factory = {
    'veri': VeRi,
    'naic' :NAIC,
    'contest':CONTEST,
    'contest2019':CONTEST2019
}

def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, _, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids

def val_collate_fn(batch):
    imgs, pids, camids, img_paths = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids, img_paths


def make_dataloader(cfg):
    if cfg.DATASETS.HARD_AUG:
        train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            # T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
            # RandomGaussianBlur(
            #     prob_happen=0.3,
            #     radius=10
            # ),
            #RandomPatch和erasing可以考虑只使用其中一个
            RandomPatch(
                prob_happen=0.3),
            #     # patch_max_area=cfg.INPUT.RANDOM_PATCH_AREA
            #     ),
            T.transforms.RandomAffine(0, translate=None, scale=[0.9, 1.1], shear=None, resample=False, fillcolor=128),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
        train_transforms_2019 = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
            #RandomPatch和erasing可以考虑只使用其中一个
            # RandomPatch(
            #     prob_happen=0.3),
            #     # patch_max_area=cfg.INPUT.RANDOM_PATCH_AREA
            #     ),
            T.transforms.RandomAffine(0, translate=None, scale=[0.9, 1.1], shear=None, resample=False, fillcolor=128),
            # RandomGaussianBlur(
            #     prob_happen=cfg.INPUT.GAUSSIAN_BLUR_PROB,
            #     radius=cfg.INPUT.GAUSSIAN_BLUR_RADIUS
            # ),
            RandomPatch(
                prob_happen=0.3),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)

            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    else:
        train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
            # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    num_classes = dataset.num_train_pids

    # train_set = ImageDataset(dataset.train[:68673], train_transforms) + \
    #     ImageDataset(dataset.train[68673:], train_transforms_2019)

    train_set = ImageDataset(dataset.train, train_transforms)

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn
        )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))


    val_set_green = ImageDataset(dataset.query_green + dataset.gallery_green, val_transforms)
    val_loader_green = DataLoader(
        val_set_green, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    val_set_normal = ImageDataset(dataset.query_normal + dataset.gallery_normal, val_transforms)
    val_loader_normal = DataLoader(
        val_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, val_loader_green, val_loader_normal, len(dataset.query_green), len(dataset.query_normal), num_classes
    # return train_loader, num_classes
    # val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    # val_loader = DataLoader(
    #     val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
    #     collate_fn=val_collate_fn
    # )
    # return train_loader, val_loader, len(dataset.query), num_classes


def make_dataloader_Pseudo(cfg):
    if cfg.DATASETS.HARD_AUG:
        train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
            T.transforms.RandomAffine(0, translate=None, scale=[0.9, 1.1], shear=None, resample=False, fillcolor=128),
            T.ToTensor(),
            #这个均值和方差也得考虑去掉
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    else:
        train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    num_classes = dataset.num_train_pids

    train_set = ImageDataset(dataset.train, train_transforms)

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn
        )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))


    val_set_green = ImageDataset(dataset.query_green + dataset.gallery_green, val_transforms)
    val_loader_green = DataLoader(
        val_set_green, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    return train_loader, val_loader_green, len(dataset.query_green), num_classes, dataset, train_set, train_transforms
