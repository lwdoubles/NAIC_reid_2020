from .bases import BaseImageDataset
import os.path as osp
import os
from collections import defaultdict

class CONTEST2019(BaseImageDataset):
    def __init__(self, root='../contest'):
    # def __init__(self, root='../REID2019fusai'):
        super(CONTEST2019, self).__init__()
        self.dataset_dir = root
        # self.dataset_dir_train = osp.join(self.dataset_dir, 'train/images')
        self.dataset_dir_train = osp.join(self.dataset_dir, 'train2019')
        self.dataset_dir_test = osp.join(self.dataset_dir, 'image_A')

        train = self.process_dir(self.dataset_dir_train, relabel=True)
        query_green, query_normal = self._process_dir_test(self.dataset_dir_test,  query = True)
        gallery_green, gallery_normal = self._process_dir_test(self.dataset_dir_test, query = False)


        self.train = train
        self.query_green = query_green
        self.gallery_green = gallery_green
        self.query_normal = query_normal
        self.gallery_normal = gallery_normal
        # print("the len of query is {}\n".format(len(query)))
        # print("the len of gallery is {}\n".format(len(gallery)))
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
    
    def process_dir(self, data_dir, relabel=True):
        filename = osp.join(data_dir, 'label.txt')
        dataset = []
        camid = 1
        count_image = defaultdict(list)
        with open(filename, 'r') as  file_to_read:
            while True:
                lines = file_to_read.readline()
                if not lines:
                    break
                img_name, img_label = [i for i in lines.split(':')]
                count_image[img_label].append(img_name)
        val_imgs = {}
        pid_container = set()
        for pid, img_name in count_image.items():
            if len(img_name) < 2:
                pass
            else:
                val_imgs[pid] = count_image[pid]
                pid_container.add(pid)
        #将label转成0,1,2这样的形式    
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        for pid, img_name in val_imgs.items():
            pid = pid2label[pid]
            for img in img_name:
                dataset.append((osp.join(data_dir, img), pid, camid))
        return dataset
    
    def _process_dir_test(self, data_dir, query=True):
        if query:
            subfix = 'query'
        else:
            subfix = 'gallery'

        datatype = ['green', 'normal']
        for index, type in enumerate(datatype):
            filename = osp.join(data_dir, '{}_{}.txt'.format(subfix, type))
            dataset = []
            with open(filename, 'r') as file_to_read:
                while True:
                    lines = file_to_read.readline()
                    if not lines:
                        break
                    for i in lines.split():
                        img_name = i

                    dataset.append((osp.join(self.dataset_dir_test, subfix, img_name), 1, 1))
            if index == 0:
                dataset_green = dataset
        return dataset_green, dataset