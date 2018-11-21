# Back2Future: Unsupervised Learning of Multi-Frame Optical Flow with Occlusions
The code for [Unsupervised Learning of Multi-Frame Optical Flow with Occlusions](http://www.cvlibs.net/publications/Janai2018ECCV.pdf). 

We propose a framework for **unsupervised learning of optical flow and occlusions over multiple frames**. Our multi-frame, occlusion-sensitive formulation outperforms existing unsupervised two-frame methods and even produces results on par with some fully supervised methods.

More details can be found on our [Project Page](https://avg.is.tuebingen.mpg.de/research_projects/back2future).

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
computeFlow = back2future.init('Ours-Soft-ft-KITTI')
```
We provide three pre-trained models from our paper. Download and store the models into the [models](models) folder: 
- [Ours-Hard](https://www.dropbox.com/s/6h7xrj2zn986o9l/RoamingImages_H.t7?dl=0)
- [Ours-Soft-ft-KITTI](https://www.dropbox.com/s/q441yclt066jf0i/RoamingImages_H_KITTI_S.t7?dl=0)
- [Ours-Soft-ft-Sintel](https://www.dropbox.com/s/x90pv7d2v6wu7gb/RoamingImages_H_Sintel_S.t7?dl=0)

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
More details in [flowExtensions](flowExtensions.lua).

<a name="training"></a>
## Training
We provide the following files to read in images and gt flow from RoamingImages, KITTI and Sintel:<br>
**NOTE**: Replace [PATH] in each file with the root path of the corresponding dataset.
- datasets/RoamingImages.dat: Our [RoamingImages dataset](https://www.dropbox.com/s/tfz3zc64an0mizh/RoamingImages.tar?dl=0)
- datasets/Kitti2015.dat: [KITTI 2015 Multiview Training](http://www.cvlibs.net/download.php?file=data_scene_flow_multiview.zip) set (excluding frames 9-12)
- datasets/Sintel.dat: [Sintel Clean + Final Training](http://sintel.is.tue.mpg.de/)


Pre-training using the hard constraint network on RoamingImages with linear motion:
```bash
th main.lua -cache checkpoints -expName Hard_Constraint -dataset RoamingImages -ground_truth \
-pme 1 -pme_criterion OBCC -smooth_flow 2
```

Fine-tuning 'Hard_Constraint' model after 10 iterations using the soft constraint network on KITTI:
```bash
th main.lua -cache checkpoints -expName Soft_KITTI -dataset Kitti2015 -LR 0.00001 \
-pme 2 -pme_criterion OBGCC -pme_alpha 0 -pme_beta 1 -pme_gamma 1 \
-smooth_flow 0.1 -smooth_second_order -const_vel 0.0001 -past_flow -convert_to_soft \
-retrain checkpoints/Hard_Constraint/model_10.t7 -optimState checkpoints/Hard_Constraint/optimState_10.t7
```

Fine-tuning 'Hard_Constraint' model after 10 iterations using the soft constraint network on Sintel:
```bash
th main.lua -cache checkpoints -expName Soft_Sintel -dataset Sintel -ground_truth -LR 0.00001 \
-pme 4 -pme_criterion OBGCC -pme_alpha 1 -pme_beta 0 -pme_gamma 0 \
-smooth_flow 0.1 -smooth_second_order -const_vel 0.0001 -past_flow -convert_to_soft \
-retrain checkpoints/Hard_Constraint/model_10.t7 -optimState checkpoints/Hard_Constraint/optimState_10.t7
```
The complete list of parameters and the default values can be found in [opts.lua](opts.lua).

We also provide our pre-trained model on RoamingImages 'Ours-Hard' and the corresponding state of the optimizer for individual fine-tuning:
[[model](https://www.dropbox.com/s/6h7xrj2zn986o9l/RoamingImages_H.t7?dl=0)], [[state](https://www.dropbox.com/s/91shuecs2aevbij/RoamingImages_H_optimState.t7?dl=0)]
```bash
-retrain [PATH]/RoamingImages_H.t7 -optimState [PATH]/RoamingImages_H_optimState.t7
```

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
Free for non-commercial and scientific research purposes. For commercial use, please contact ps-license@tue.mpg.de. <br>
Check LICENSE file for details.

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
