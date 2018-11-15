-- Copyright 2018 Joel Janai, Fatma Güney, Anurag Ranjan and the Max Planck Gesellschaft.
-- All rights reserved.
-- This software is provided for research purposes only.
-- By using this software you agree to the terms of the license file
-- in the root folder.
-- For commercial use, please contact ps-license@tue.mpg.de.

paths.dofile('myCmdLine.lua')

local M = { }

unpack = table.unpack

function M.parse(arg)
    local cmd = torch.myCmdLine()
    cmd:text()
    cmd:text('Back2Future: Unsupervised Learning of Multi-Frame Optical Flow with Occlusions')
    cmd:text()
    cmd:text('Options:')

    ------------ General options --------------------
    cmd:option('-expName',      'debug', 'Experiment name')
    cmd:option('-debug',        0, '1: turn off debug mode')
    cmd:option('-cache',        '/is/rg/avg/jjanai/projects/unsupervised_multiframe_flow/experiments/checkpoints', 'subdirectory in which to save/log experiments')
    cmd:option('-dataset',      'flying_chairs_training', 'File name of dataset')
    cmd:option('-manualSeed',   2, 'Manually set RNG seed')
    cmd:option('-GPU',          1, 'Default preferred GPU')
    cmd:option('-nGPU',         1, 'Number of GPUs to use by default')
    cmd:option('-parallel',     'data', 'Parallize: data | network')
    cmd:option('-backend',      'cudnn',  'Options: cudnn | ccn2 | cunn')
    cmd:option('-resubmit',     0,  'stores one checkpoint and restarts')

    ------------- Data options ------------------------
    cmd:option('-nDonkeys',        4, 'number of donkeys to initialize (data loading threads)')
    cmd:option('-scale',           1,  'scale input before cropping')
    cmd:option('-fineWidth',       128, 'the length of the fine flow field, data is at res. 256 x 128')
    cmd:option('-fineHeight',      64, 'the width of the fine flow field')
    cmd:option('-cropWidth',       0, 'the length of the fine flow field, data is at res. 256 x 128')
    cmd:option('-cropHeight',      0, 'the width of the fine flow field')
    cmd:option('-gaussian_noise',  0.0, 'provide path to an optimState to reload from')
    cmd:option('-normalize_images',     1, 'provide path to an optimState to reload from')
    cmd:option('-rand_crop',            1, 'provide path to an optimState to reload from')

    ------------- Training options --------------------
    cmd:option('-augment',          0,    'augment the data')
    cmd:option('-nEpochs',          1000, 'Number of total epochs to run')
    cmd:option('-epochSize',        1000, 'Number of batches per epoch')
    cmd:option('-epochStore',       5,    'Store every X epoch')
    cmd:option('-batchSize',        32,   'mini-batch size (1 = pure stochastic)')
    cmd:option('-epochNumber',      1,    'Manual epoch number (useful on restarts)')
    cmd:option('-retrain',          'none', 'provide path to model to retrain with')
    cmd:option('-optimState',       'none', 'provide path to an optimState to reload from')
    cmd:option('-cont',             false, 'if true, will set epochNumber, retrain and optimState to the latest model saved.')
    cmd:option('-convert_to_backward_flow',  false, 'Convert old model to new model')

    ------------- Training/Criterion options --------------------
    cmd:option('-optimize',       'pme',  'epe (supervised) or pme (unsupervised)')
    cmd:option('-sizeAverage',    false,  'if true, div by number of pixels for all losses')
    cmd:option('-ground_truth',    true,  'use ground truth for error')
    cmd:option('-backward_flow',  false, 'Jointly predict forward and backward flow')
    cmd:option('-pme_criterion',  'OBCC', 'BCC, OBCC, SSIM, OSSIM')
    cmd:option('-pme_penalty',    'L1',   'L1 or L2 or Lorentzian')
    cmd:option('-pme_alpha',    1,   'OBGCC brightness weight')
    cmd:option('-pme_beta',    1,   'OBGCC gradient in x dir weight')
    cmd:option('-pme_gamma',    1,   'OBGCC gradient in y dir weight')
    cmd:option('-smooth_flow_penalty', 'L1',   'L1 or L2 or Lorentzian')
    cmd:option('-smooth_occ_penalty', 'Quadratic',   'Quadratic, L1 or Lorentzian or Dirac or KL')
    cmd:option('-epe', 1.0,   'Weight reconstruction loss')
    cmd:option('-pme', 10.0,   'Weight reconstruction loss')
    cmd:option('-smooth_second_order', false,   'Use second order smoothness')
    cmd:option('-smooth_flow', 1.0,   'Weight flow smoothness loss')
    cmd:option('-const_vel', 1.0,   'Weight flow smoothness loss')
    cmd:option('-smooth_occ', 1.0,   'Weight occlusion smoothness loss')
    cmd:option('-prior_occ', 1.0,   'Weight occlusion gmm loss')
    cmd:option('-mask_entropy', 0,   'Weight temporal consistency loss')
    cmd:option('-tce', 0,   'Weight temporal consistency loss')
    cmd:option('-tce_occ', 0,   'Weight temporal consistency loss')
    
    ---------- Optimization options ----------------------
    cmd:option('-LR',    0.0, 'learning rate; if set, overrides default LR/WD recipe')
    cmd:option('-momentum',        0.9,  'momentum')
    cmd:option('-weightDecay',     0.0, 'weight decay')
    cmd:option('-optimizer',       'adam', 'adam or sgd')

    ---------- Model options ----------------------------------
    cmd:option('-L1', 'models/modelL1_4.t7', 'Trained Level 1 model')
    cmd:option('-L2', 'models/modelL2_4.t7', 'Trained Level 2 model')
    cmd:option('-L3', 'models/modelL3_4.t7', 'Trained Level 3 model')
    cmd:option('-L4', 'models/modelL4_4.t7', 'Trained Level 4 model')

    cmd:option('-netType',     'pwc', 'Lua network file, unet or volcon (spynet) or pwc')
    cmd:option('-frames',          3,    'number of frames in symmetric window')  
    cmd:option('-two_frame',      0,    'use only two frame for predictions')  
    cmd:option('-no_occ',     false,  'turn off occlusion reasoning')
    cmd:option('-dis_occ',     1,  'use discrete occlusion vars')
    
    cmd:option('-levels',  6, 'number of warping levels')
    cmd:option('-residual',  1, '1: use residual flow')
    cmd:option('-flow_input',  1, '1: input upsampled flow to next outer level')
    cmd:option('-mask_input',  0, '1: input upsampled flow to next outer level')
    cmd:option('-occ_input',  0, '1: input upsampled flow to next outer level')
    cmd:option('-rescale_flow',  0, 'provide path to an optimState to reload from')
    cmd:option('-flownet_factor',  20, 'provide path to an optimState to reload from')
    
    cmd:option('-original_pwc',  0, 'provide path to an optimState to reload from')
    cmd:option('-pwc_ws',      5, 'window size for cost volume (pwc)')
    cmd:option('-pwc_skip',    2, '0: full resolution image, >0: skip highest resolutions')
    cmd:option('-pwc_siamese',    1, '0: use images directly, 1: extract features with siamese network (pwc)')
    cmd:option('-pwc_sum_cvs',    false, 'Sum forward and backward cost volume if enabled')
    
    cmd:option('-unet_levels',  5, 'number of levels in unet')
    cmd:option('-unet_pyramid_loss',  0, 'loss on every level of unet_level 1')
    cmd:text()

    local opt = cmd:parse(arg or {})
    opt.save = paths.concat(opt.cache)
    if opt.expName == '' then
      opt.expName = os.date('%Y%m%d_%H%M%S')
    end
    opt.save = paths.concat(opt.save, '' .. opt.expName)
    print('Saving everything to: ' .. opt.save)
    os.execute('mkdir -p ' .. opt.save)
    
    if opt.no_occ == true then
      opt.pwc_sum_cvs = true
    end
    
    if opt.resubmit > 0 then
      opt.cont = true
    end

    opt.frames = tonumber(opt.frames)
    assert(opt.frames == 2 or (opt.frames % 2) ~= 0)
    opt.channels = 3 * opt.frames

    if opt.dataset == 'flying_chairs_training' then
      opt.loadSize = {opt.channels, 384, 512}
      opt.fineWidth = 512
      opt.fineHeight = 384
      
      if opt.original_pwc == 1 then
        opt.loadSize = {opt.channels, 384, 448}
        opt.fineWidth = 448
        opt.fineHeight = 384
        opt.levels = 7
        opt.pwc_skip = 2
        opt.pwc_ws = 9
        opt.pwc_full_res = 0
        opt.residual = 0
        opt.flownet_factor = 20
        opt.rescale_flow = 0
        opt.sizeAverage = false
        opt.two_frame = 0
        print('Setting PWC original parameters!!')
      end
    elseif opt.dataset == 'flying_things_finalpass_training' then
      opt.loadSize = {opt.channels, 540, 960}
      opt.fineWidth = 960
      opt.fineHeight = 540
      opt.cropWidth = 640
      opt.cropHeight = 320
      opt.scale = 0.7
    elseif string.match(opt.dataset, 'Cityscapes') then
      opt.loadSize = {opt.channels, 1024, 2048}
      opt.fineWidth = 2048 
      opt.fineHeight = 1024
      opt.cropWidth = 640
      opt.cropHeight = 320
    elseif string.match(opt.dataset, 'Kitti') then
      opt.loadSize = {opt.channels, 375, 1242}
      opt.fineWidth = 1242  -- 1242
      opt.fineHeight = 375 -- 375
      opt.cropWidth = 640    -- 640 CROPPED FOR FASTER TRAINING
      opt.cropHeight = 320
    elseif string.match(opt.dataset, 'Sintel') then
      opt.loadSize = {opt.channels, 436, 1024}
      opt.fineWidth = 1024  -- 1242
      opt.fineHeight = 436 -- 375
      opt.cropWidth = 640    -- 640 CROPPED FOR FASTER TRAINING
      opt.cropHeight = 384
    elseif opt.dataset == 'toy_NOISE_training' then
      opt.loadSize = {opt.channels, 32, 64}
      opt.fineWidth = 64
      opt.fineHeight = 32
    elseif opt.dataset == 'toy_training' then
      opt.loadSize = {opt.channels, 32, 64}
      opt.fineWidth = 64
      opt.fineHeight = 32
    elseif string.match(opt.dataset, 'slowflow') or string.match(opt.dataset, 'HD') then
      opt.loadSize = {opt.channels, 320, 640}
      opt.fineWidth = 640
      opt.fineHeight = 320
      opt.batchSize = 8
    elseif string.match(opt.dataset, 'DEBUG') then
      opt.loadSize = {opt.channels, 640, 1280}
      opt.scale = 0.5
      opt.fineWidth = 640
      opt.fineHeight = 320
      opt.batchSize = 8
    else
      opt.loadSize = {opt.channels, 64, 128}
    end
    
    if opt.cropWidth > 0 and opt.cropHeight > 0 then
      opt.loadSize = {opt.channels, opt.cropHeight, opt.cropWidth}
      opt.fineWidth = opt.cropWidth
      opt.fineHeight = opt.cropHeight
    else
      opt.fineWidth = opt.fineWidth * opt.scale
      opt.fineHeight = opt.fineHeight * opt.scale
    end
    
    if string.match(opt.dataset, 'slowflow')  then
      opt.rand_crop = 1
    end

    if opt.optimize == 'epe' then
      opt.ground_truth = true
    end

    -- log parameters
    cmd:log(opt.save .. '/log', opt)

    return opt
end

return M
