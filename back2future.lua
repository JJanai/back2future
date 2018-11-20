----------------------------------------------------
---- MULTI FRAME PWC Net
-----------------------------------------------------
-- Multi frame PWC Net model
--
-- input -> N images (batchSize x N * ChannelSize x Height x Width)
-- output -> table consisting of flow_future, flow_past, occlusion, warped_img_1, ..., warped_img_N from finest to coarest level
--  flow_x -> flow to future or past (batchSize x 2 x Height x Width)
--  occlusion -> occlusions (batchSize x 2 x Height x Width)
--  warped_img_i -> warped image i (batchSize x ChannelSize x Height x Width)
--
-- Copyright 2018 Joel Janai, Fatma Güney, Anurag Ranjan and the Max Planck Gesellschaft.
-- All rights reserved.
-- This software is provided for research purposes only.
-- By using this software you agree to the terms of the license file
-- in the root folder.
---------------------------------------------------------------

require 'image'
local TF = require 'transforms'
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'stn'
require 'spy'
require 'models.CostVolMulti'
local flowX = require 'flowExtensions'

local M = {}

local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}

local mean = meanstd.mean
local std = meanstd.std
local occ_threshold = 0.6666

local function normalize(imgs)
  return TF.ColorNormalize(meanstd)(imgs)
end
M.normalize = normalize

local computeFlow = function(im1, im2, im3)
    local imgs = torch.cat({im1, im2, im3}, 1)
    imgs = TF.ColorNormalize(meanstd)(imgs)
  
    local width = imgs:size(3)
    local height = imgs:size(2)
    
    -- width and height should be divisibly by 2^6 using 7 levels 
    local fineWidth 
    local fineHeight
    if width%64 == 0 then
      fineWidth = width
    else
      fineWidth = width - math.fmod(width, 64)
    end
  
    if height%64 == 0 then
      fineHeight = height
    else
      fineHeight = height - math.fmod(height, 64)
    end  

    print(fineWidth, fineHeight)
         
    imgs = image.scale(imgs, fineWidth, fineHeight)
  
    imgs = imgs:resize(1,9,fineHeight,fineWidth):cuda()
    local est = model:forward(imgs)
  
    -- get flow from table
    flow_est = est[1]:squeeze():double()

    -- resize and scale flow
    local sc_h = height/flow_est:size(2)
    local sc_w = width/flow_est:size(3)
    flow_est = image.scale(flow_est, width, height, 'simple')
    flow_est[2] = flow_est[2]*sc_h
    flow_est[1] = flow_est[1]*sc_w
  
    -- get occlusions from table
    occ_est = est[3]:squeeze():double()
    fwd_occ_est = torch.ge(occ_est[{{2},{},{}}], occ_threshold)  -- Future occlusions
    fwd_occ_est = image.scale(fwd_occ_est, width, height, 'simple')
    bwd_occ_est = torch.ge(occ_est[{{1},{},{}}], occ_threshold)  -- Past occlusions
    bwd_occ_est = image.scale(bwd_occ_est, width, height, 'simple')
  
    return flow_est, fwd_occ_est, bwd_occ_est
  
  end

local function init(opt)
    opt = opt or 'Ours-Soft-ft-KITTI'
  
    if opt=="Ours-Hard" then
        modelPath = paths.concat('models', 'RoamingImages_H.t7')
    end
  
    if opt=="Ours-Hard-ft-KITTI" then
        modelPath = paths.concat('models', 'RoamingImages_H_KITTI_H.t7')
    end
  
    if opt=="Ours-Hard-ft-Sintel" then
        modelPath = paths.concat('models', 'RoamingImages_H_Sintel_H.t7')
    end
  
    if opt=="Ours-Soft-ft-KITTI" then
        modelPath = paths.concat('models', 'RoamingImages_H_KITTI_S.t7')
    end
  
    if opt=="Ours-Soft-ft-Sintel" then
        modelPath = paths.concat('models', 'RoamingImages_H_Sintel_S.t7')
    end

    -- read in model
    model = torch.load(modelPath)
    if torch.type(model) == 'nn.DataParallelTable' then
        model = model:get(1)
    end
    model:evaluate()

    -- compute number of frames
    local nw_modules = model:findModules('nn.Narrow')
    local cv_modules = model:findModules('nn.CostVolMulti')
    if #cv_modules > 0 then
        channels = 3 * #nw_modules
    else
        channels = model:parameters()[1]:size(2)
    end
  
    return computeFlow
  end
  M.init = init
  
  
  
  return M
