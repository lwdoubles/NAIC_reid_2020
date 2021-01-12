# NAIC_reid_2020 初赛17，复赛21
## 方案说明
### 划分数据集
执行下列命令划分数据集
`python divided_dataset.py --data_dir_query image_B/query --data_dir_gallery image_B/gallery --save_dir image-B/`
### 训练相应模型：
`python train.py --config_file configs/naic_10_9_2.yml`
### 测试相应模型并生成json文件：
`python test.py --config_file configs/naic_10_9_2.yml`
### 模型融合:
`python ensemble_dist.py`
