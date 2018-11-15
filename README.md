# Back2Future: Unsupervised Learning of Multi-Frame Optical Flow with Occlusions
This code is based on the paper [Unsupervised Learning of Multi-Frame Optical Flow with Occlusions](http://www.cvlibs.net/publications/Janai2018ECCV.pdf). 

Overview:
* [Setup](#setUp)
* [Usage](#usage) 
* [Training](#training) 
* [Optical Flow Utilities](#flowUtils) 
* [References](#references)

<a name="setUp"></a>
## Setup
You need to have [Torch.](http://torch.ch/docs/getting-started.html#_)

Install other required packages
```bash
cd extras/spybhwd
luarocks make
cd ../stnbhwd
luarocks make
```
<a name="usage"></a>
## Usage
#### Set up back2future
```lua
back2future = require('back2future')
computeFlow = back2future.setup()
```
#### Load images and compute flow
```lua
im1 = image.load('samples/00001_img1.ppm' )
im2 = image.load('samples/00001_img2.ppm' )
flow = computeFlow(im1, im2)
```
To save your flow fields to a .flo file use [flowExtensions.writeFLO](#writeFLO).

<a name="training"></a>
## Training
Training sequentially is faster than training end-to-end since you need to learn small number of parameters at each level. To train a level `N`, we need the trained models at levels `1` to `N-1`. You also initialize the model with a pretrained model at `N-1`.

E.g. To train level 3, we need trained models at `L1` and `L2`, and we initialize it  `modelL2_3.t7`.
```bash
th main.lua -fineWidth 128 -fineHeight 96 -level 3 -netType volcon \
-cache checkpoint -data FLYING_CHAIRS_DIR \
-L1 models/modelL1_3.t7 -L2 models/modelL2_3.t7 \
-retrain models/modelL2_3.t7
```

<a name="flowUtils"></a>
## Optical Flow Utilities
We provide `flowExtensions.lua` containing various functions to make your life easier with optical flow while using Torch/Lua. You can just copy this file into your project directory and use if off the shelf.
```lua
flowX = require 'flowExtensions'
```
#### [flow_magnitude] flowX.computeNorm(flow_x, flow_y)
Given `flow_x` and `flow_y` of size `MxN` each, evaluate `flow_magnitude` of size `MxN`.

#### [flow_angle] flowX.computeAngle(flow_x, flow_y)
Given `flow_x` and `flow_y` of size `MxN` each, evaluate `flow_angle` of size `MxN` in degrees.

#### [rgb] flowX.field2rgb(flow_magnitude, flow_angle, [max], [legend])
Given `flow_magnitude` and `flow_angle` of size `MxN` each, return an image of size `3xMxN` for visualizing optical flow. `max`(optional) specifies maximum flow magnitude and `legend`(optional) is boolean that prints a legend on the image.

#### [rgb] flowX.xy2rgb(flow_x, flow_y, [max])
Given `flow_x` and `flow_y` of size `MxN` each, return an image of size `3xMxN` for visualizing optical flow. `max`(optional) specifies maximum flow magnitude.

#### [flow] flowX.loadFLO(filename)
Reads a `.flo` file. Loads `x` and `y` components of optical flow in a 2 channel `2xMxN` optical flow field. First channel stores `x` component and second channel stores `y` component.

<a name="writeFLO"></a>
#### flowX.writeFLO(filename,F)
Write a `2xMxN` flow field `F` containing `x` and `y` components of its flow fields in its first and second channel respectively to `filename`, a `.flo` file.

#### [flow] flowX.loadPFM(filename)
Reads a `.pfm` file. Loads `x` and `y` components of optical flow in a 2 channel `2xMxN` optical flow field. First channel stores `x` component and second channel stores `y` component.

#### [flow_rotated] flowX.rotate(flow, angle)
Rotates `flow` of size `2xMxN` by `angle` in radians. Uses nearest-neighbor interpolation to avoid blurring at boundaries.

#### [flow_scaled] flowX.scale(flow, sc, [opt])
Scales `flow` of size `2xMxN` by `sc` times. `opt`(optional) specifies interpolation method, `simple` (default), `bilinear`, and `bicubic`.

#### [flowBatch_scaled] flowX.scaleBatch(flowBatch, sc)
Scales `flowBatch` of size `Bx2xMxN`, a batch of `B` flow fields by `sc` times. Uses nearest-neighbor interpolation.

<a name="references"></a>
## References
1. Our code is based on [anuragranj/spynet.](https://github.com/anuragranj/spynet)
2. The warping code is based on [qassemoquab/stnbhwd.](https://github.com/qassemoquab/stnbhwd)
3. The images in `samples` are from Flying Chairs dataset: 
   A. Dosovitskiy, et al. "Flownet: Learning optical flow with convolutional networks." 2015 IEEE International Conference on Computer Vision (ICCV). IEEE, 2015.
4. Some parts of `flowExtensions.lua` are adapted from [marcoscoffier/optical-flow](https://github.com/marcoscoffier/optical-flow/blob/master/init.lua) with help from [fguney](https://github.com/fguney).
   
## License
Free for non-commercial and scientific research purposes. For commercial use, please contact ps-license@tue.mpg.de. Check LICENSE file for details.

## When using this code, please cite
J. Janai, F. Gney, A. Ranjan, M. Black, and A. Geiger. "Unsupervised learning of multi-frame optical flow with occlusions." In Proc. of the European Conf. on Computer Vision (ECCV), 2018. 
