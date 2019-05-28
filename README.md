## Description
face verification

## Requirements
1. [DarkerNet](https://github.com/isLouisHsu/DarkerNet);

## Usage
### Install
``` shell
mkdir build && cd build
cmake .. && make
```

### Run
``` shell
./mobilefacenet
./mobilefacenet --image1 xxx.jpg --image2 xxx.jpg
./mobilefacenet --datasets lfw --minface 36
./mobilefacenet --datasets lfw --aligned
```

## Model
![graph](/images/graph_run=.png)

## Details
1. Input size: $(3, 112 ,96)$;
2. `Global Depthwise Convolutional Layer` was replaced by `Locally Connected Layer`;
3. Download aligned(112x96) LFW images from [Align-LFW@BaiduDrive](https://pan.baidu.com/s/1r6BQxzlFza8FM8Z8C_OCBg).

## Reference
1. [MobileFaceNet_TF](https://github.com/sirius-ai/MobileFaceNet_TF)
