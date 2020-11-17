# Few-shot Object Detection via Feature Reweighting (Remake Version)

This repo is the remake version for the original code of https://github.com/bingykang/Fewshot_Detection .

If any problems, open an issue!

## What I Have Done:

- This code has been upgraded to work with python3.6+, pytorch1.2+ which is more suitable for nowadays settings.
- Reorganize the code and add some annotation to understand the code.
- Modify the ouput information, which is more simple and efficient.


## Additional Details:

### dataset dir
In this code, you have to prepare dataset as the original code did. Please remember to change the root dir of text file in the ```./data/``` path, such as ```voc_traindict_bbox_1shot.txt```.

### result of k-shot k=1,2,5,10
the base training part is always the same for different k, but the finetune part is different. To check the result of k-shot(k=1,2,5,10), you have to find the file: ./cfg/metatune.data

find the code ```meta = data/voc_traindict_bbox_5shot.txt``` and change the num 5 to 1 or others.

### why proposal is decreased to 0?
Don't worry! It will increase after several epochs. 

### memory usage / cuda out of memory
Since the detection part is based on yolov2, multi-scale training is used. After the beginning few epochs(maybe 20), the input shape will change from (416,416) to (512,512) or random shape. Choose an adaptive batch size is very important!.




## Original Readme 
Implementation for the paper:

[Few-shot Object Detection via Feature Reweighting](https://arxiv.org/abs/1812.01866), ICCV 2019

[Bingyi Kang](https://scholar.google.com.sg/citations?user=NmHgX-wAAAAJ)\*, [Zhuang Liu](https://liuzhuang13.github.io)\*, [Xin Wang](https://people.eecs.berkeley.edu/~xinw/), [Fisher Yu](https://www.yf.io), [Jiashi Feng](https://sites.google.com/site/jshfeng/home) and [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/) (\* equal contribution)

Our code is based on  [https://github.com/marvis/pytorch-yolo2](https://github.com/marvis/pytorch-yolo2) and developed with  Python 2.7 & PyTorch 0.3.1.



### Training our model on VOC

- ``` $PROJ_ROOT : project root ```
- ``` $DATA_ROOT : dataset root ```

#### Prepare dataset
+ Get The Pascal VOC Data
```
cd $DATA_ROOT
wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xf VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar
```

+ Generate Labels for VOC
```
wget http://pjreddie.com/media/files/voc_label.py
python voc_label.py
cat 2007_train.txt 2007_val.txt 2012_*.txt > voc_train.txt
```

+ Generate per-class Labels for VOC (used for meta inpput)
```
cp $PROJ_ROOT/scripts/voc_label_1c.py $DATA_ROOT
cd $DATA_ROOT
python voc_label_1c.py
```

+ Generate few-shot image list
To use our few-shot datasets
```
cd $PROJ_ROOT
python scripts/convert_fewlist.py 
```

You may want to generate new few-shot datasets
Change the ''DROOT'' varibale in scripts/gen_fewlist.py to $DATA_ROOT
```
python scripts/gen_fewlist.py # might be different with ours
```

#### Base Training
+ Modify Cfg for Pascal Data
Change the data/metayolo.data file 
```
metayolo=1
metain_type=2
data=voc
neg = 1
rand = 0
novel = data/voc_novels.txt             // file contains novel splits
novelid = 0                             // which split to use
scale = 1
meta = data/voc_traindict_full.txt
train = $DATA_ROOT/voc_train.txt
valid = $DATA_ROOT/2007_test.txt
backup = backup/metayolo
gpus=1,2,3,4
```

+ Download Pretrained Convolutional Weights
```
wget http://pjreddie.com/media/files/darknet19_448.conv.23
```

+ Train The Model
```
python train_meta.py cfg/metayolo.data cfg/darknet_dynamic.cfg cfg/reweighting_net.cfg darknet19_448.conv.23
```

+ Evaluate the Model
```
python valid_ensemble.py cfg/metayolo.data cfg/darknet_dynamic.cfg cfg/reweighting_net.cfg path/toweightfile
python scripts/voc_eval.py results/path/to/comp4_det_test_
```

#### Few-shot Tuning
+ Modify Cfg for Pascal Data
Change the data/metatune.data file 
```
metayolo=1
metain_type=2
data=voc
tuning = 1
neg = 0
rand = 0
novel = data/voc_novels.txt                 
novelid = 0
max_epoch = 2000
repeat = 200
dynamic = 0
scale=1
train = $DATA_ROOT/voc_train.txt
meta = data/voc_traindict_bbox_5shot.txt
valid = $DATA_ROOT/2007_test.txt
backup = backup/metatune
gpus  = 1,2,3,4
```


+ Train The Model
```
python train_meta.py cfg/metatune.data cfg/darknet_dynamic.cfg cfg/reweighting_net.cfg path/to/base/weightfile
```

+ Evaluate the Model
```
python valid_ensemble.py cfg/metatune.data cfg/darknet_dynamic.cfg cfg/reweighting_net.cfg path/to/tuned/weightfile
python scripts/voc_eval.py results/path/to/comp4_det_test_
```

### Citation
```
@inproceedings{kang2019few,
  title={Few-shot Object Detection via Feature Reweighting},
  author={Kang, Bingyi and Liu, Zhuang and Wang, Xin and Yu, Fisher and Feng, Jiashi and Darrell, Trevor},
  booktitle={ICCV},
  year={2019}
}
```
