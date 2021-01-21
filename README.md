### Requirements

+ python 3.6
+ Pytorch>=0.4.1

### Dataset

https://pan.baidu.com/s/1PhSViqjxR6J2RfYnSmgFlw

提取码：qftc

直接放置根目录即可，data/images/resized目录下的数据集需解压

### Train

`CUDA_VISIBLE_DEVICES=0,1 python Train.py --basic_model=VisualAttention --use_MIA=True --iteration_times=2`

根据下游任务不同basic_model可选[VisualAttention, ConceptAttention, VisualCondition, ConceptCondition, VisualRegionalAttention]

### Test

`CUDA_VISIBLE_DEVICES=0 python Test.py  --basic_model=basic_model_name --use_MIA=True --iteration_times=2`

### Reference

```
@inproceedings{Liu2019MIA,
  author    = {Fenglin Liu and
               Yuanxin Liu and
               Xuancheng Ren and
               Xiaodong He and
               Xu Sun},
  title     = {Aligning Visual Regions and Textual Concepts for Semantic-Grounded
               Image Representations},
  booktitle = {NeurIPS},
  pages     = {6847--6857},
  year      = {2019}
}
```

+ https://github.com/lancopku/simNet