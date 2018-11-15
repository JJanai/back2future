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
local flowX = require 'flowExtensions'

local M = {}

local eps = 1e-6
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}

local mean = meanstd.mean
local std = meanstd.std

local function normalize(imgs)
  return TF.ColorNormalize(meanstd)(imgs)
end
M.normalize = normalize

local computeFlow = function(im1, im2)
    local imgs = torch.cat(im1, im2, 1)
    imgs = TF.ColorNormalize(meanstd)(imgs)
  
    local width = imgs:size(3)
    local height = imgs:size(2)
    
    local fineWidth, fineHeight
    
    if width%32 == 0 then
      fineWidth = width
    else
      fineWidth = width + 32 - math.fmod(width, 32)
    end
  
    if height%32 == 0 then
      fineHeight = height
    else
      fineHeight = height + 32 - math.fmod(height, 32)
    end  
         
    imgs = image.scale(imgs, fineWidth, fineHeight)
  
    imgs = imgs:resize(1,6,fineHeight,fineWidth):cuda()
    local flow_est = model:forward(imgs)
  
    -- get flow from table
    flow_est = flow_est:squeeze():float()

    -- resize and scale flow
    local sc_h = height/flow_est:size(2)
    local sc_w = width/flow_est:size(3)
    flow_est = image.scale(flow_est, w, h, 'simple')
    flow_est[2] = flow_est[2]*sc_h
    flow_est[1] = flow_est[1]*sc_w
  
    return flow_est
  
  end

local function init(opt)
    opt = opt or 'Ours-Hard-ft-KITTI'
    
    if opt=="Classic" then
        modelPath = paths.concat('models', 'modelL1_F.t7')
    end
    
    if opt=="Multi" then
        modelPath = paths.concat('models', 'modelL1_F.t7')
    end

    if opt=="Ours-None" then
        modelPath = paths.concat('models', 'modelL1_F.t7')
    end
  
    if opt=="Ours-Soft" then
        modelPath = paths.concat('models', 'modelL1_C.t7')
    end
  
    if opt=="Ours-Hard" then
        modelPath = paths.concat('models', 'modelL1_4.t7')
    end
  
    if opt=="Ours-Hard-ft-KITTI" then
        modelPath = paths.concat('models', 'modelL1_3.t7')
    end
  
    if opt=="Ours-Hard-ft-Sintel" then
        modelPath = paths.concat('models', 'modelL1_K.t7')
    end
    
    model = torch.load(modelPath)
    if torch.type(model) == 'nn.DataParallelTable' then
        model = model:get(1)
    end
    model:evaluate()
  
    return easyComputeFlow
  end
  M.easy_setup = easy_setup
  
  
  
  return M
