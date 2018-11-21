# Back2Future: Unsupervised Learning of Multi-Frame Optical Flow with Occlusions
This code is based on the paper [Unsupervised Learning of Multi-Frame Optical Flow with Occlusions](http://www.cvlibs.net/publications/Janai2018ECCV.pdf). 

Overview:
* [Setup](#setUp)
* [Usage](#usage) 
* [Training](#training) 
* [Optical Flow Utilities](#flowUtils) 
* [License](#license)
* [Please cite us](#license)

<a name="setUp"></a>
## Setup
You need to have [Torch.](http://torch.ch/docs/getting-started.html#_)
<br>

The code was tested with Torch7, CUDA 9.0, cudnn 7.0. When using CUDA 9.0 you will run into [problems](https://github.com/torch/torch7/issues/1133) following the Torch installation guide. Execute the following command before calling install.sh to resolve the problem:
```bash
export TORCH_NVCC_FLAGS="-D__CUDA_NO_HALF_OPERATORS__"
```

For cudnn 7.0, you will also need to clone and install the Revision 7 branch of the cudnn.torch repository:
```bash
git clone https://github.com/soumith/cudnn.torch -b R7
cd cudnn.torch
luarocks make cudnn-scm-1.rockspec
```

Install other required packages:
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
computeFlow = back2future.init()
```
#### Load images and compute flow
```lua
im1 = image.load('samples/frame_0009.png' )
im2 = image.load('samples/frame_0010.png' )
im3 = image.load('samples/frame_0011.png' )
flow, fwd_occ, bwd_occ  = computeFlow(im1, im2, im3)
```
#### Storing flow field, flow visualization and forward occlusions
```lua
flowX = require('flowExtensions')
flowX.writeFLO('samples/flow.flo', flow:float())

floImg = flowX.xy2rgb(flow[{1,{},{}}], flow[{2,{},{}}])
image.save('samples/flow.png', floImg)

image.save('samples/fwd_occ.png', fwd_occ * 255)
image.save('samples/bwd_occ.png', bwd_occ * 255)
```
More details in [flowExtensions](#flowUtils).

<a name="training"></a>
## Training
We provide the following files to read in images and gt flow from RoamingImages, KITTI and Sintel:<br>
**NOTE**: Replace [PATH] in each file with the root path of the corresponding dataset.
- [RoamingImages.dat](datasets/RoamingImages.dat): Our RoamingImages dataset
- [Kitti2015.dat](datasets/Kitti2015.dat): KITTI 2015 Multiview Training set (excluding frames 9-12)
- [Sintel.dat](datasets/Sintel.dat): Sintel Clean + Final Training


Pre-training using the hard constraint network on RoamingImages with linear motion:
```bash
th main.lua -cache checkpoints -expName Hard_Constraint -dataset RoamingImages \
-frames 3 -netType pwc -levels 7 \
-optimize pme -pme 1 -pme_criterion OBCC -pme_penalty L1 \
-smooth_occ 0.1 -prior_occ 0.1 -smooth_flow 2 \
-batchSize 8 -nDonkeys 8 -nGPU 1
```

Fine-tuning 'Hard_Constraint' model after 10 iterations using the soft constraint network on KITTI:
```bash
th main.lua -cache checkpoints -netType pwc -expName Soft_KITTI -dataset Kitti2015 \
-frames 3 -netType pwc -levels 7 \
-optimize pme -pme 2 -pme_criterion OBGCC -pme_penalty L1 \
-pme_alpha 0 -pme_beta 1 -pme_gamma 1 \
-smooth_occ 0.1 -prior_occ 0.1 -smooth_flow 0.1 -smooth_second_order \
-const_vel 0.0001 -past_flow -convert_to_soft \
-retrain checkpoints/Hard_Constraint/model_10.t7 -optimState checkpoints/Hard_Constraint/optimState_10.t7 \
-batchSize 8 -nDonkeys 8 -nGPU 1 -LR 0.00001
```

Fine-tuning 'Hard_Constraint' model after 10 iterations using the soft constraint network on Sintel:
```bash
th main.lua -cache checkpoints -netType pwc -expName Soft_KITTI -dataset Sintel -ground_truth \
-frames 3 -netType pwc -levels 7 \
-optimize pme -pme 4 -pme_criterion OBGCC -pme_penalty L1 \
-pme_alpha 1 -pme_beta 0 -pme_gamma 0 \
-smooth_occ 0.1 -prior_occ 0.1 -smooth_flow 0.1 -smooth_second_order \
-const_vel 0.0001 -past_flow -convert_to_soft \
-retrain checkpoints/Hard_Constraint/model_10.t7 -optimState checkpoints/Hard_Constraint/optimState_10.t7 \
-batchSize 8 -nDonkeys 8 -nGPU 1 -LR 0.00001

```
A complete list of options can be found in [opts.lua](opts.lua).

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
3. The images in `samples` are from KITTI 2015 dataset: <br>
   A. Geiger, P. Lenz,  C.  Stiller, R. Urtasun: "Vision  meets  robotics:  The  KITTI  dataset." International Journal of Robotics Research (IJRR). (2013)<br>
   M. Menze, A. Geiger: "Object scene flow for autonomous vehicles." In: Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR). (2015)<br>
4. Some parts of `flowExtensions.lua` are adapted from [marcoscoffier/optical-flow](https://github.com/marcoscoffier/optical-flow/blob/master/init.lua) with help from [fguney](https://github.com/fguney).
   
<a name="license"></a>
## License
Free for non-commercial and scientific research purposes. For commercial use, please contact ps-license@tue.mpg.de. Check LICENSE file for details.

## When using this code, please cite

@inproceedings{Janai2018ECCV,<br>
&nbsp;&nbsp;title = {Unsupervised Learning of Multi-Frame Optical Flow with Occlusions },<br>
&nbsp;&nbsp;author = {Janai, Joel and G{\"u}ney, Fatma and Ranjan, Anurag and Black, Michael J. and Geiger, Andreas},<br>
&nbsp;&nbsp;booktitle = {European Conference on Computer Vision (ECCV)},<br>
&nbsp;&nbsp;volume = {Lecture Notes in Computer Science, vol 11220},<br>
&nbsp;&nbsp;pages = {713--731},<br>
&nbsp;&nbsp;publisher = {Springer, Cham},<br>
&nbsp;&nbsp;month = sep,<br>
&nbsp;&nbsp;year = {2018},<br>
&nbsp;&nbsp;month_numeric = {9}<br>
}
