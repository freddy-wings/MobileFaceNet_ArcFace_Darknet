## Description
Face verification.

## Requirements
1. [DarkerNet](https://github.com/isLouisHsu/DarkerNet)
2. [OpenCV3.4.6](https://github.com/opencv/opencv)

## Usage
### Install
``` shell
mkdir build && cd build
cmake .. && make
```
### Run
1. Download dataset from [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/)
2. Prepare `lfw/`, `veiw2/pairs`, `lfw_112x96`([Align-LFW@BaiduDrive](https://pan.baidu.com/s/1r6BQxzlFza8FM8Z8C_OCBg))

```
data/
├── lfw/
├── lfw_112x96/
└── pairs.txt
```

``` shell
./mobilefacenet --help
```

## Model
![graph](/images/graph_run=.png)

## Prepare data
Download `lfw` and `CASIA-WebFace`, extract to `data/`

``` shell
cd pretrained/prepare_data/
```

1. detect `CASIA-WebFace`
    ``` shell
    python detect.py -d casia
    ```
    generate file `data/CASIA_detect.txt`

    ``` shell
    python detect.py -d lfw
    ```
    generate file `data/lfw_detect.txt`

2. crop and aligned
    ``` shell
    python crop.py -d casia
    ```
    generate directory `data/CASIA-WebFace-Aligned` and `CASIA-WebFace-Unaligned`

    ``` shell
    python crop.py -d lfw
    ```
    generate directory `data/lfw-Aligned` and `lfw-Unaligned`

3. generate label file
    ``` shell
    python label.py
    ```
    generate file `data/CASIA_label_train.txt` and `data/CASIA_label_valid.txt`


```
data/
├── CASIA_detect.txt
├── CASIA_label_train.txt
├── CASIA_label_valid.txt
├── CASIA-WebFace/
├── CASIA-WebFace-Aligned/
└── CASIA-WebFace-Unaligned/
```

## Training
``` shell
cd pretrained/train/
python main.py
```
model saved as `pretrained/train/ckpt/MobileFacenet_xxxx.pkl`

## Testing
``` shell
cd pretrained/train/
python test_lfw.py
```

## Extract Weights
``` shell
cd pretrained/
python extract_weights_cfg.py -f mobilefacenet.pkl
```
generate `weights/mobilefacenet.weights` and `cfg/mobilefacenet.cfg`.

## Details
1. Input size: (3, 112 ,96);
2. `Global Depthwise Convolutional Layer` is replaced by `Locally Connected Layer`;
3. Crop and align: [Face-Detection-MTCNN](https://louishsu.xyz/2019/05/05/Face-Detection-MTCNN/);

## Results

``` shell
./mobilefacenet --dataset lfw --minface 36 --thresh 0.3
python static.py
```

![statistic_lfw](/images/statistic_lfw.png)
``` shell
Reading LFW...OK
[6000]/[6000] >> Elapsed 3273.3s, FPS 1.99
Gt: 1  Pred: 1  Cosine: 0.461
Total: 6000 | Bad:    0 | Detected: 6000
Accuracy: 0.99 | Precision: 0.99 | Recall: 0.98
```


## Reference
1. [sirius-ai/MobileFaceNet_TF - Github](https://github.com/sirius-ai/MobileFaceNet_TF)
2. [wy1iu/sphereface - Github](https://github.com/wy1iu/sphereface)
3. [Xiaoccer/MobileFaceNet_Pytorch - Github](https://github.com/Xiaoccer/MobileFaceNet_Pytorch)
4. [Chen-Zhihui/ cvn/src/Cvutil/src/cp2tform.cpp - Github](https://github.com/Chen-Zhihui/cvn/blob/093672ed4a890ce6bd240c51a068bca8a3597bde/src/Cvutil/src/cp2tform.cpp)