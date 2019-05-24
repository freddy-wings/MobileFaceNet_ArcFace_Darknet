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
```

## Model
![graph](/images/graph_run=.png)

## Details
1. Input size: $(3, 112 ,96)$;
2. `Global Depthwise Convolutional Layer` was replaced by `Locally Connected Layer`;

## Reference
1. [MobileFaceNet_TF](https://github.com/sirius-ai/MobileFaceNet_TF)
