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
    cmd:option('-expName',      'exp',    'Experiment name')
    cmd:option('-debug',        0,        'Turn on/off debug mode')
    cmd:option('-cache',        'checkpoints',            'Subdirectory in which to save/log experiments')
    cmd:option('-dataset',      'RoamingImages', 'File name of dataset')
    cmd:option('-ground_truth',  false,  'Dataset file contains ground truth path (e.g. Sintel)')
    cmd:option('-manualSeed',   2,        'Manually set RNG seed')
    cmd:option('-GPU',          1,        'Default preferred GPU')
    cmd:option('-nGPU',         1,        'Number of GPUs to use by default')
    cmd:option('-backend',      'cudnn',  'Options: cudnn | ccn2 | cunn')

    ------------- Data options ------------------------
    cmd:option('-nDonkeys',        4,   'Number of donkeys to initialize (data loading threads)')
    cmd:option('-scale',           1,   'Scale input before cropping')
    cmd:option('-fineWidth',       128, 'Width of the fine flow field, data is at res. 256 x 128')
    cmd:option('-fineHeight',      64,  'Height of the fine flow field')
    cmd:option('-rand_crop',       1,   'Randomly crop input images')
    cmd:option('-cropWidth',       0,   'Width of crop during training')
    cmd:option('-cropHeight',      0,   'Height of crop during training')
    cmd:option('-gaussian_noise',  0.0, 'Std of zero mean gaussian noise (0: deactivated)')
    cmd:option('-normalize_images',  1, 'Normalize images by mean and std')

    ------------- Training options --------------------
    cmd:option('-augment',          0,    'Turn on/off data augmentation')
    cmd:option('-nEpochs',          1000, 'Number of total epochs to run')
    cmd:option('-epochSize',        1000, 'Number of batches per epoch')
    cmd:option('-epochStore',       1,    'Store every X epoch')
    cmd:option('-batchSize',        32,   'Mini-batch size (1 = pure stochastic)')
    cmd:option('-epochNumber',      1,    'Manual epoch number (useful on restarts)')
    cmd:option('-retrain',          'none', 'Provide path to model to retrain with')
    cmd:option('-optimState',       'none', 'Provide path to an optimState to reload from')
    cmd:option('-cont',             false, 'Set epochNumber, retrain and optimState to the latest model saved.')
    cmd:option('-convert_to_soft',  false, 'Convert Hard Constraint model to Soft Constraint model')

    ------------- Training/Criterion options --------------------
    cmd:option('-optimize',       'pme',  'epe (supervised) or pme (unsupervised)')
    cmd:option('-sizeAverage',    false,  'Normalize all losses by number of pixels')
    cmd:option('-past_flow',  false,  'Jointly predict future and past flow (Soft Constraint)')
    
    cmd:option('-epe', 0.0,           'Weight epe loss')
    cmd:option('-pme', 1.0,           'Weight reconstruction loss')
    cmd:option('-pme_criterion',      'OBCC', 'BCC, OBCC, SSIM, OSSIM')
    cmd:option('-pme_penalty',        'L1',   'Quadratic or L1 or Lorentzian')
    cmd:option('-pme_alpha',    1,    'OBGCC brightness weight')
    cmd:option('-pme_beta',    1,     'OBGCC gradient in x dir weight')
    cmd:option('-pme_gamma',    1,    'OBGCC gradient in y dir weight')
    cmd:option('-smooth_flow', 1.0,   'Weight flow smoothness loss')
    cmd:option('-smooth_second_order', false,   'Use second order smoothness')
    cmd:option('-smooth_flow_penalty', 'L1',   'Quadratic or L1 or Lorentzian')
    cmd:option('-smooth_occ_penalty', 'Quadratic', 'Quadratic, L1 or Lorentzian or Dirac or KL')
    cmd:option('-smooth_occ', 1.0,    'Weight occlusion smoothness loss')
    cmd:option('-prior_occ', 1.0,     'Weight occlusion prior loss')
    cmd:option('-const_vel', 1.0,     'Weight for constant velocity loss (Soft Constraint)')
    
    ---------- Optimization options ----------------------
    cmd:option('-LR',             0.0,    'learning rate; if set, overrides default LR/WD recipe')
    cmd:option('-momentum',       0.9,    'momentum')
    cmd:option('-weightDecay',    0.0,    'weight decay')
    cmd:option('-optimizer',      'adam', 'adam or sgd')

    ---------- Model options ----------------------------------
    cmd:option('-netType',      'pwc',  'Lua network file, unet or spynet or pwc')
    cmd:option('-frames',       3,      'number of frames in symmetric window')  
    cmd:option('-two_frame',    0,      'use only two frame for predictions')  
    cmd:option('-no_occ',       false,  'turn off occlusion reasoning')
    
    cmd:option('-levels',  6,           'Number of warping levels')
    cmd:option('-residual',  0,         '1: Use residual flow')
    cmd:option('-flow_input',  1,       '1: input upsampled flow to next outer level')
    cmd:option('-occ_input',  0,        '1: input upsampled occlusions to next outer level')
    cmd:option('-rescale_flow',  0,     'provide path to an optimState to reload from')
    cmd:option('-flownet_factor',  20,  'provide path to an optimState to reload from')
    
    cmd:option('-original_pwc',  0,       'Use the orignal PCW-Net')
    cmd:option('-pwc_ws',      9,         'Window size for cost volumes')
    cmd:option('-pwc_skip',    2,         '0: full resolution image, >0: skip highest resolutions')
    cmd:option('-pwc_siamese',    1,      '0: use images directly, 1: extract features with siamese network (pwc)')
    cmd:option('-pwc_sum_cvs',    false,  'Sum future and past cost volume')

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

    opt.frames = tonumber(opt.frames)
    assert(opt.frames == 2 or (opt.frames % 2) ~= 0)
    opt.channels = 3 * opt.frames

    if opt.dataset == 'flying_chairs' then
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
    else
      opt.loadSize = {opt.channels, 320, 640}
      opt.fineWidth = 640
      opt.fineHeight = 320
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
