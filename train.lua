-- Copyright 2018 Joel Janai, Fatma Güney, Anurag Ranjan and the Max Planck Gesellschaft.
-- All rights reserved.
-- This software is provided for research purposes only.
-- By using this software you agree to the terms of the license file
-- in the root folder.
-- For commercial use, please contact ps-license@tue.mpg.de.
--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'optim'
require 'image'
paths.dofile('myLogger.lua')

--[[
   1. Setup SGD optimization state and learning rate schedule
   2. Create loggers.
   3. train - this function handles the high-level training loop,
              i.e. load data, train model, save model and state to disk
   4. trainBatch - Used by train() to train a single batch after the data is loaded.
]]--

-- Setup a reused optimization state (for sgd). If needed, reload it from disk
local optimState = {
  learningRate = opt.LR,
  learningRateDecay = 0.0,
  momentum = opt.momentum,
  dampening = 0.0,
  weightDecay = opt.weightDecay
}

if opt.optimState ~= 'none' then
  local retrain_opt = opt.optimState
  assert(paths.filep(retrain_opt), 'File not found: ' .. retrain_opt)
  print('Loading optimState from file: ' .. retrain_opt)
  optimState = torch.load(retrain_opt)
end

-- Learning rate annealing schedule. We will build a new optimizer for
-- each epoch.
--
-- By default we follow a known recipe for a 55-epoch training. If
-- the learningRate command-line parameter has been specified, though,
-- we trust the user is doing something manual, and will use her
-- exact settings for all optimization.
--
-- Return values:
--    diff to apply to optimState,
--    true IFF this is the first epoch of a new regime


local level_weights = {
  0.005, 0.01, 0.02, 0.08, 0.32, 0.64, 1.28
}

if opt.sizeAverage then
  level_weights = {
    1, 1, 1, 1, 1, 1
  }
end

local function paramsForEpoch(epoch)
  -- PWC parameters and schedule
  local LR = 1e-4
  if opt.LR > 0 then
    LR = opt.LR
  end
  
  local WD = 0
  if opt.weightDecay > 0 then
    WD = opt.weightDecay
  end
  
  local regimes = {
    -- start, end,       LR,   WD,
    { 1,   200,   LR,   WD },
    { 201, 400,   LR/2, WD },
    { 401, 600,   LR/4, WD },
    { 601, 800,   LR/8, WD },
    { 801, 1e3,   LR/16, WD },
  }

  for _, row in ipairs(regimes) do
    if epoch >= row[1] and epoch <= row[2] then
      return { learningRate=row[3], weightDecay=row[4] }, epoch >= row[1]
    end
  end
end

-- 2. Create loggers.
trainLogger = optim.myLogger(paths.concat(opt.save, 'train.log'))
local batchNumber
local loss_epoch
local avg_epe
local avg_epe_nocc
local avg_epe_occ
local avg_oacc
local avg_occ_acc_bwd
local avg_occ_acc_vis
local avg_occ_acc_fwd

-- 3. train - this function handles the high-level training loop,
--            i.e. load data, train model, save model and state to disk
function train()
  print('==> doing epoch on training data:')
  print("==> online epoch # " .. epoch)

  local params, newRegime = paramsForEpoch(epoch)
  if newRegime then
    optimState = {
      learningRate = params.learningRate,
      learningRateDecay = 0.0,
      momentum = opt.momentum,
      dampening = 0.0,
      weightDecay = params.weightDecay
    }
  end
  batchNumber = 0
  cutorch.synchronize()

  -- set the dropouts to training mode
  model:training()

  local tm = torch.Timer()
  loss_epoch = 0
  avg_epe = 0
  avg_epe_nocc = 0
  avg_epe_occ = 0
  avg_oacc = 0
  avg_occ_acc_bwd = 0
  avg_occ_acc_vis = 0
  avg_occ_acc_fwd = 0
  for i = 1, opt.epochSize do
    -- queue jobs to data-workers
    donkeys:addjob(
      -- the job callback (runs in data-worker thread)
      function()
        local inputs, labels, masks = trainLoader:sample(opt.batchSize)
        return inputs, labels, masks
      end,
      -- the end callback (runs in the main thread)
      trainBatch
    )
  end

  donkeys:synchronize()
  cutorch.synchronize()

  loss_epoch = loss_epoch / opt.epochSize
  avg_epe = avg_epe / opt.epochSize
  avg_epe_nocc = avg_epe_nocc / opt.epochSize
  avg_epe_occ = avg_epe_occ / opt.epochSize
  avg_oacc = avg_oacc / opt.epochSize
  avg_occ_acc_bwd = avg_occ_acc_bwd / opt.epochSize
  avg_occ_acc_vis = avg_occ_acc_vis / opt.epochSize
  avg_occ_acc_fwd = avg_occ_acc_fwd / opt.epochSize

  if opt.ground_truth == true then
    trainLogger:add{['avg epe (train set)'] = avg_epe, ['avg epe non occ (train set)'] = avg_epe_nocc, ['avg epe occ (train set)'] = avg_epe_occ, ['avg loss (train set)'] = loss_epoch,['avg occ acc (train set)'] = avg_oacc,
    ['avg bwd acc (train set)'] = avg_occ_acc_bwd,['avg vis acc (train set)'] = avg_occ_acc_vis,['avg fwd acc (train set)'] = avg_occ_acc_fwd}
    print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
        .. 'average loss (per batch): %.2f \t average epe (per batch): %.2f \t average epe non occ (per batch): %.2f \t average epe occ (per batch): %.2f \t average occ acc (per batch): %.2f (%.2f,%.2f,%.2f)',
        epoch, tm:time().real, loss_epoch, avg_epe, avg_epe_nocc, avg_epe_occ, avg_oacc, avg_occ_acc_bwd, avg_occ_acc_vis, avg_occ_acc_fwd))
  else
    trainLogger:add{['avg loss (train set)'] = loss_epoch}
    print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
        .. 'average loss (per batch): %.2f \t ',
        epoch, tm:time().real, loss_epoch))
  end
  print('\n')

  -- save model
  collectgarbage()

  -- clear the intermediate states in the model before saving to disk
  -- this saves lots of disk space
  model:clearState()
  if epoch == 1 or epoch % opt.epochStore == 0 then
    saveDataParallel(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model) -- defined in util.lua
    torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)
  end
end -- of train()
-------------------------------------------------------------------------------------------
-- GPU inputs (preallocate)
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()
local masks = torch.CudaTensor()

local timer = torch.Timer()
local dataTimer = torch.Timer()

local parameters, gradParameters = model:getParameters()

-- 4. trainBatch - Used by train() to train a single batch after the data is loaded.
function trainBatch(inputsCPU, labelsCPU, masksCPU)
  cutorch.synchronize()
  collectgarbage()
  local dataLoadingTime = dataTimer:time().real
  timer:reset()

  -- transfer over to GPU
  inputs:resize(inputsCPU:size()):copy(inputsCPU)
  labels:resize(labelsCPU:size()):copy(labelsCPU)
  masks:resize(masksCPU:size()):copy(masksCPU)

  local err = 0
  local occ = 0
  local epe = 0
  local epe_nocc = 0
  local epe_occ = 0
  local oacc = 0
  local occ_acc_bwd = 0
  local occ_acc_fwd = 0
  local occ_acc_vis = 0
  local pme = 0
  local sflow = 0
  local entropy = 0
  local socc = 0
  local gocc = 0
  local tflow = 0
  local tocc = 0
  local outputs
  feval = function(x)

    local out_warp_start, n_unit_out, n_flow, ref_c
    if opt.frames == 2 then
      ref_c = 1
      out_warp_start = 2
      n_unit_out = 2 -- only flow and warped
      n_flow = 1
    else
      -- idx to ref
      local ref = 0.5 * (opt.frames + 1)
      ref_c = (ref - 1) * 3 + 1
      out_warp_start = 3
      n_unit_out = opt.frames + 1 -- flow + occ + warped
      n_flow = 1
      
      if opt.past_flow then
        n_flow = 2
        n_unit_out = n_unit_out + 1
        out_warp_start = 4
      end
    end

    -- ################################### forward model ###################################
    model:zeroGradParameters()
    outputs = model:forward(inputs[{{},{1,opt.frames*3},{},{}}]:contiguous())

    if opt.debug == 1 then
      for i = 1,opt.frames do
        local b = 1
--        for b=1,inputsCPU:size(1) do
          require 'image'

          if i < opt.frames then
            local img = torch.Tensor(3,outputs[out_warp_start+i-1]:size(3),outputs[out_warp_start+i-1]:size(4))
            img:copy(outputs[out_warp_start+i-1][{{b},{},{},{}}])
            local mx = torch.max(img)
            local mn = torch.min(img)
            img = (img - mn) / (mx - mn)
            image.save(string.format("tmp/%d_frame_%03d_warp.jpg", batchNumber, i), img)
          end

          img = torch.Tensor(3,inputsCPU:size(3),inputsCPU:size(4))
          img:copy(inputsCPU[{{b},{(i - 1)* 3 + 1,(i - 1)* 3 + 3},{},{}}][1])
          local mx = torch.max(img)
          local mn = torch.min(img)
          img = (img - mn) / (mx - mn)
          image.save(string.format("tmp/%d_frame_%03d_ref.jpg", batchNumber, i), img)
--        end
      end
    end

    local gradOutputs = {}
    -- DOWNSAMPLE
    local down = nn.SpatialAveragePooling(2,2,2,2):cuda()
    local down_nn = nn.SpatialAveragePooling(1,1,2,2):cuda()
    local down_sampled = inputs:clone()
    local down_sampled_flow = labels[{{},{1,2},{},{}}]:clone()
    local down_sampled_occ = labels[{{},{3},{},{}}]:clone()
    local down_sampled_mask = masks:clone()

    local levels = #outputs / n_unit_out 
    
    for f = 1, #outputs do
      table.insert(gradOutputs, torch.CudaTensor(outputs[f]:size()):zero())
    end

    -- ################################### SUPERVISION ###################################
    if opt.optimize == 'epe' then
      for l = 0, (levels-1) do
        if l > 0 then
          down_sampled_flow = down_nn:forward(down_sampled_flow):clone()
          down_sampled_mask = down_nn:forward(down_sampled_mask):clone()
          if opt.rescale_flow == 1 then
            down_sampled_flow:div(2)
          end
          if opt.frames > 2 and not opt.no_occ then
            down_sampled_occ = down_nn:forward(down_sampled_occ):clone()
          end
        end

        --              1-4, 5-8, 9-12
        local sub_outs = {unpack(outputs, l * n_unit_out + 1, (l+1) * n_unit_out)}

        -- Flow Supervised Loss
        local err_f = opt.epe * criterion:forward(sub_outs[1], {down_sampled_flow, down_sampled_mask})
        err = err + err_f * level_weights[l+1]
        gradOutputs[l * n_unit_out + 1]:add(criterion:backward(sub_outs[1], {down_sampled_flow, down_sampled_mask}):clone():mul(opt.epe * level_weights[l+1]))
          
        if opt.frames > 2 then
          if not opt.no_occ then
            -- Occlusion Supervised Loss
            local occ_repeated = down_sampled_occ
            
            -- convert gt occlusions
            local tmp1 = occ_repeated[{{},{1},{},{}}]
            local tmp2 = occ_repeated[{{},{2},{},{}}]
            occ_repeated[{{},{1},{},{}}] = torch.eq(tmp1,0):float() + 0.5*torch.eq(tmp1,0.5):float()
            occ_repeated[{{},{2},{},{}}] = torch.eq(tmp2,1):float() + 0.5*torch.eq(tmp2,0.5):float()
            occ_repeated = occ_repeated:cuda()
  
            local tmp = level_weights[l+1] * occ_criterion:forward(sub_outs[out_warp_start-1], occ_repeated)
            err = err + tmp
            occ = occ + tmp
            gradOutputs[l * n_unit_out + out_warp_start-1]:add(occ_criterion:backward(sub_outs[out_warp_start-1], occ_repeated):clone():mul(level_weights[l+1]))
          end
        end
      end
    end

    -- highest res epe
    if opt.ground_truth == true then
      -- Flow Supervised Loss
      local epe_b = criterion:forward(outputs[1] * opt.flownet_factor, {labels[{{},{1,2},{},{}}] * opt.flownet_factor, masks})
      if opt.sizeAverage == false then
        epe_b = epe_b / masks:sum()
      end
      epe = epe + epe_b
        
      local lbl_occ = labels[{{},{4},{},{}}]:squeeze():float()
      local norm
      
      -- epe in visible regions!
      local occ = lbl_occ:ne(0.5):cudaByte()
      local vis_epe_map = criterion.epe_map:clone()
      vis_epe_map = vis_epe_map:maskedFill(occ, 0)
      norm = (1-occ):float():cmul(masksCPU):sum()
      vis_epe_map = 0
      if norm > 0 then
        vis_epe_map = vis_epe_map / norm
        if opt.flownet_factor ~= 1 then
          vis_epe_map = vis_epe_map * opt.flownet_factor
        end
        epe_nocc = epe_nocc + vis_epe_map
      end
      
      -- epe in occluded regions!
      local vis = lbl_occ:eq(0.5):cudaByte()
      local occ_epe_map = criterion.epe_map:clone()  -- DONT USE CRITERION IN BETWEEN
      occ_epe_map = occ_epe_map:maskedFill(vis, 0)
      norm = (1-vis):float():cmul(masksCPU):sum()
      occ_epe_map = 0
      if norm > 0 then
        occ_epe_map = occ_epe_map / norm
        if opt.flownet_factor ~= 1 then
          occ_epe_map = occ_epe_map * opt.flownet_factor
        end
        epe_occ = epe_occ + occ_epe_map
      end

      local tmp, occ_est_sharp, occ_map
      if opt.frames > 2 and (not opt.no_occ) then
        if outputs[out_warp_start-1]:size(2) == 1 then
          tmp = outputs[out_warp_start-1]:float():squeeze()
          occ_est_sharp = torch.mul(tmp, 2):round():div(2)
        elseif outputs[out_warp_start-1]:size(2) == 3 then
          _,tmp = torch.max(outputs[out_warp_start-1],2)
          tmp = tmp:float():squeeze()
          occ_est_sharp = torch.div(tmp - 1, 2)
        else
          occ_est_sharp = torch.round((1 - outputs[out_warp_start-1][{{},{1},{},{}}]) + (outputs[out_warp_start-1][{{},{2},{},{}}])):mul(0.5)
          occ_est_sharp = occ_est_sharp:float()
        end
        
        
        local lbl_occ = labels[{{},{3},{},{}}]:squeeze():float()
        occ_map = torch.eq(lbl_occ, occ_est_sharp):float()
        oacc = oacc + (occ_map:sum() / lbl_occ:nElement())
    
        local bwd_occ = lbl_occ:eq(0)
        norm = bwd_occ:sum()
        if norm > 0 then
          occ_acc_bwd = occ_acc_bwd + torch.eq(occ_est_sharp, lbl_occ):maskedSelect(bwd_occ):float():sum() / norm
        end
        
        local vis = lbl_occ:eq(0.5)
        norm = vis:sum()
        if norm > 0 then
          occ_acc_vis = occ_acc_vis + torch.eq(occ_est_sharp, lbl_occ):maskedSelect(vis):float():sum() / norm
        end
        
        local fwd_occ = lbl_occ:eq(1)
        norm = fwd_occ:sum()
        if norm > 0 then
          occ_acc_fwd = occ_acc_fwd + torch.eq(occ_est_sharp, lbl_occ):maskedSelect(fwd_occ):float():sum() / norm
        end
      end
    end

    -- ################################### PHOTOMETRIC LOSS AND SMOOTHNESS ###################################
    if(opt.optimize == 'pme') then
      for l = 0, (levels-1) do
        if l > 0 then
          down_sampled = down:forward(down_sampled)
        end
        --              1-4, 5-8, 9-12
        local sub_outs = {unpack(outputs, l * n_unit_out + 1, (l+1) * n_unit_out)}
        
        pme_criterion.pwc_flow_scaling = model.flow_scale[levels - l]
        
        -- Flow Smoothness Loss
        for i = 1, n_flow do
          sflow = sflow + level_weights[l+1] * opt.smooth_flow * fs_criterion:forward(sub_outs[i], down_sampled[{{},{ref_c,ref_c+2},{},{}}])
          local tmp = level_weights[l+1] * opt.smooth_flow * fs_criterion:backward(sub_outs[i], down_sampled[{{},{ref_c,ref_c+2},{},{}}]):clone()
          gradOutputs[l * n_unit_out + i]:add(tmp)
        end
        fs_criterion:clear()
        
        -- constant velocity loss
        if opt.past_flow then
          sflow = sflow + level_weights[l+1] * opt.const_vel * cv_criterion:forward(sub_outs)
          local tmp = cv_criterion:backward(sub_outs)
          gradOutputs[l * n_unit_out + 1]:add(level_weights[l+1] * opt.const_vel * tmp[1])
          gradOutputs[l * n_unit_out + 2]:add(level_weights[l+1] * opt.const_vel * tmp[2])
        end
        
        -- Photometric Loss
        pme = pme + level_weights[l+1] * opt.pme * pme_criterion:forward(sub_outs, down_sampled[{{},{ref_c,ref_c+2},{},{}}])
        local grads = pme_criterion:backward(sub_outs, down_sampled[{{},{ref_c,ref_c+2},{},{}}])
        for i,v in ipairs(grads) do
          local tmp = level_weights[l+1] * opt.pme * v:clone()
          if opt.frames == 2 then
            gradOutputs[l * n_unit_out + out_warp_start + i - 1]:add(tmp)
          else
            gradOutputs[l * n_unit_out + out_warp_start + i - 2]:add(tmp)
          end
        end
        pme_criterion:clear()
      
        if opt.frames > 2 and (not opt.no_occ) then
          -- Occlusion Smoothness Loss
          if(opt.smooth_occ > 0) then
            socc = socc + level_weights[l+1] * opt.smooth_occ * os_criterion:forward(sub_outs[out_warp_start-1], down_sampled[{{},{ref_c,ref_c+2},{},{}}])
            gradOutputs[l * n_unit_out + out_warp_start - 1]:add(level_weights[l+1] * opt.smooth_occ, os_criterion:backward(sub_outs[out_warp_start-1], down_sampled[{{},{ref_c,ref_c+2},{},{}}]):clone())
            os_criterion:clear()
          end

          -- Occlusion Prior Loss
          if(opt.prior_occ > 0) then
            gocc = gocc + level_weights[l+1] * opt.prior_occ * oprior_criterion:forward(sub_outs[out_warp_start-1], down_sampled[{{},{ref_c,ref_c+2},{},{}}])
            gradOutputs[l * n_unit_out + out_warp_start - 1]:add(level_weights[l+1] * opt.prior_occ, oprior_criterion:backward(sub_outs[out_warp_start-1], down_sampled[{{},{ref_c,ref_c+2},{},{}}]):clone())
          end
        end
        
        collectgarbage()
      end

      err = pme + sflow + entropy + socc + gocc
    end

    err = err + tflow + tocc

    -- ################################### BACKPROP ###################################
    model:backward(inputs, gradOutputs)

    return err, gradParameters
  end

  if opt.optimizer == 'adam' then
    optim.adam(feval, parameters, optimState)
  elseif opt.optimizer == 'sgd' then
    optim.sgd(feval, parameters, optimState)
  else
    error("Specify Optimizer")
  end

  -- DataParallelTable's syncParameters
  if model.needsSync then
    model:syncParameters()
  end

  cutorch.synchronize()
  batchNumber = batchNumber + 1
  loss_epoch = loss_epoch + err
  avg_epe = avg_epe + epe
  avg_epe_nocc = avg_epe_nocc + epe_nocc
  avg_epe_occ = avg_epe_occ + epe_occ
  avg_oacc = avg_oacc + oacc
  avg_occ_acc_bwd = avg_occ_acc_bwd + occ_acc_bwd
  avg_occ_acc_vis = avg_occ_acc_vis + occ_acc_vis
  avg_occ_acc_fwd = avg_occ_acc_fwd + occ_acc_fwd

  -- Calculate top-1 error, and print information
  if opt.optimize == 'pme' and opt.ground_truth == true then
    print(('Epoch: [%d][%d/%d]\tTime %.3f\tERR %.3f\tPME %.3f\tSmoothFlow %.3f\tSmoothOcc %.3f\tPriorOcc %.3f\t\tEPE %.3f\tEPE non Occ %.3f\tEPE Occ %.3f\tOcc Acc %.3f (%.3f,%.3f,%.3f)\tLR %.0e\tDataLoadingTime %.3f'):format(
        epoch, batchNumber, opt.epochSize, timer:time().real, err, pme, sflow, socc, gocc, epe, epe_nocc, epe_occ, oacc, occ_acc_bwd, occ_acc_vis, occ_acc_fwd,
        optimState.learningRate, dataLoadingTime))
  else
    print(('Epoch: [%d][%d/%d]\tTime %.3f Err %.4f \tOcc %.3f\tEPE %.3f\tEPE non Occ %.3f\tEPE Occ %.3f\tOcc Acc %.3f (%.3f,%.3f,%.3f)\t LR %.0e DataLoadingTime %.3f'):format(
        epoch, batchNumber, opt.epochSize, timer:time().real, err, occ, epe, epe_nocc, epe_occ, oacc, occ_acc_bwd, occ_acc_vis, occ_acc_fwd,
        optimState.learningRate, dataLoadingTime))
  end

  dataTimer:reset()
end
