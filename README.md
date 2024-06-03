# ARAI-MVSNet
The current project page provides [pytorch](https://pytorch.org/get-started/locally/) code that implements the following paper.

Title: "ARAI-MVSNet: A multi-view stereo depth estimation network with adaptive depth range and depth interval'

Abstract:

Abstract Multi-View Stereo (MVS) is a fundamental problem in geometric computer vision which aims to reconstruct a scene using multi-view images with known camera parameters. However, the mainstream approaches represent the scene with a fixed all-pixel depth range and equal depth interval partition, which will result in inadequate utilization of depth planes and imprecise depth estimation. In this paper, we present a novel multi-stage coarse-to-fine framework to achieve adaptive all-pixel depth range and depth interval. We predict a coarse depth map in the first stage, then an Adaptive Depth Range Prediction module is proposed in the second stage to zoom in the scene by leveraging the reference image and the obtained depth map in the first stage and predict a more accurate all-pixel depth range for the following stages. In the third and fourth stages, we propose an Adaptive Depth Interval Adjustment module to achieve adaptive variable interval partition for pixel-wise depth range. The depth interval distribution in this module is normalized by Z-score, which can allocate dense depth hypothesis planes around the potential ground truth depth value and vice versa to achieve more accurate depth estimation. Extensive experiments on four widely used benchmark datasets (DTU, TnT, BlendedMVS, ETH 3D) demonstrate that our model achieves state-of-the-art performance and yields competitive generalization ability. Particularly, our method achieves the highest Acc and Overall on the DTU dataset, while attaining the highest Recall and F 1-score on the Tanks and Temples intermediate and advanced dataset. Moreover, our method also achieves the lowest e 1 and e 3 on the BlendedMVS dataset and the highest Acc and F 1-score on the ETH 3D dataset, surpassing all listed methods.

## Installation

```python
pip install -r requirments.txt
```

## Testing

Download the preprocessed test data [DTU testing data](https://drive.google.com/open?id=135oKPefcPTsdtLRzoDAQtPpHuoIrpRI_) (from [Original MVSNet](https://github.com/YoYo000/MVSNet)) and unzip it as the `DTU_TESTING` folder, which should contain one `cams` folder, one `images` folder and one `pair.txt` file.

Test with the pretrained model

```python
python test.py --cfg configs/dtu.yaml
```

### dtu.yaml

Please set the following configuration

```python
OUTPUT_DIR: ""  # logfile and .pth save path 
DATA:
	TEST:
	    ROOT_DIR: "" # testing set path
	    NUM_VIEW:
	    IMG_HEIGHT:
	    IMG_WIDTH:
	    INTER_SCALE: 2.13
MODEL:
  NET_CONFIGS: "16,64,16,8"
  LAMB: 1.5
  LOSS_WEIGHTS: "0.5,1.0,1.5,2.0"
...
TEST:
  WEIGHT: "" # .pth path
  BATCH_SIZE: 1
```

## Depth Fusion

We need to apply depth fusion `tools/depthfusion.py` to get the complete point cloud. Please refer to [MVSNet](https://github.com/YoYo000/MVSNet) for more details. And use `tools/rename_ply.py` to get the rename results.
```python
python depthfusion.py

python rename_ply.py
```

To obtain the fusibile:

- Check out the modified version fusibile `git clone https://github.com/YoYo000/fusibile`
- Install fusibile by `cmake .` and `make`, which will generate the executable at `FUSIBILE_EXE_PATH`

## Evaluation
We need to download the official STL point clouds for our evaluation. Please download the [STL Point Clouds](http://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip), which is the STL reference point clouds for all the scenes. And please download the [observability masks and evaluation codes](http://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip) from the SampleSet for evaluation. 

And then use the DTU_Evaluations's code to evaluate the reconstruction results.
```
1. run BaseEvalMain_web.m
2. run ComputeStat_web.m
```

## Citation
If you find our work useful in your research, please consider citing:
```
@article{zhang2023arai,
  title={ARAI-MVSNet: A multi-view stereo depth estimation network with adaptive depth range and depth interval},
  author={Zhang, Song and Xu, Wenjia and Wei, Zhiwei and Zhang, Lili and Wang, Yang and Liu, Junyi},
  journal={Pattern Recognition},
  volume={144},
  pages={109885},
  year={2023},
  publisher={Elsevier}
}
```
