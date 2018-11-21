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
require 'nn'
require 'cunn'
require 'optim'
require 'criterions.L2Criterion'
require 'criterions.MBCCriterion'
require 'criterions.MSSIML1Criterion'
require 'criterions.OBCCriterion'
require 'criterions.OBGCCriterion'
require 'criterions.OSSIML1Criterion'
require 'criterions.SmoothnessCriterion'
require 'criterions.ConstVelCriterion'
require 'criterions.SecondOrderSmoothnessCriterion'
require 'criterions.KLDivergenceCriterion'
require 'criterions.OcclusionPriorCriterion'
require 'criterions.penalty.L1_function'
require 'criterions.penalty.quadratic_function'
require 'criterions.penalty.Lorentzian_function'

--[[
   1. Create Model
   2. Create Criterion
   3. Convert model to CUDA
]]--
local latest = 0
if opt.cont then
  latest = getLatestModelSaved(opt.save)

  if latest > 0 then
    opt.epochNumber = latest + 1
    opt.retrain = paths.concat(opt.save, 'model_' .. latest .. '.t7')
    opt.optimState = paths.concat(opt.save, 'optimState_' .. latest .. '.t7')
  end
end

-- 1. Create Network
-- 1.1 If preloading option is set, preload weights from existing models appropriately
if opt.retrain ~= 'none' then
  assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
  print('Loading model from file: ' .. opt.retrain)


  if latest <= 0 and opt.convert_to_soft then
    print("Converting hard constraint model to soft constraint model ...")
    -- CREATE A NEW MODEL
    paths.dofile('models/' .. opt.netType .. '.lua')
    print('=> Creating model from file: models/' .. opt.netType .. '.lua')
    model = createModel(opt) -- for the model creation code, check the models/ folder

    if opt.backend == 'cudnn' then
      require 'cudnn'
      cudnn.convert(model, cudnn)
    elseif opt.backend ~= 'nn' then
      error'Unsupported backend'
    end

    -- LOAD OLD MODEL
    local pre_model = loadDataParallel(opt.retrain, opt.nGPU) -- defined in util.lua

    -- if there is a model, there has to be an optimState
    assert(paths.filep(opt.optimState), 'File not found: ' .. opt.optimState)

    -- CLONE
    for m = 1, 90 do
      if pre_model.modules[m]:getParameters():nElement() > 0 then
        assert(torch.typename(model.modules[m]) == torch.typename(pre_model.modules[m]), 'Not same type!')
        assert(model.modules[m]:getParameters():nElement() == pre_model.modules[m]:getParameters():nElement(), 'Not same number of parameters!')

        for mm = 1, #model.modules[m].modules do
          if model.modules[m].modules[mm].weight then
            model.modules[m].modules[mm].weight:copy(pre_model.modules[m].modules[mm].weight)
            model.modules[m].modules[mm].bias:copy(pre_model.modules[m].modules[mm].bias)
            model.modules[m].modules[mm].gradWeight:copy(pre_model.modules[m].modules[mm].gradWeight)
            model.modules[m].modules[mm].gradBias:copy(pre_model.modules[m].modules[mm].gradBias)
          end
        end
      end
    end

    -- COPY WEIGHTS FROM FORWARD FLOW TO past flow
    local src = {30,45,60,75,90, 94, 110, 128, 146, 164}
    local dst = {93, 96, 99, 102, 105, 109, 126, 145, 164, 183}
    for m = 1, #src do
      if pre_model.modules[src[m]]:getParameters():nElement() > 0 then
        assert(torch.typename(model.modules[dst[m]]) == torch.typename(pre_model.modules[src[m]]), 'Not same type!')
        assert(model.modules[dst[m]]:getParameters():nElement() == pre_model.modules[src[m]]:getParameters():nElement(), 'Not same number of parameters!')


        for mm = 1, #model.modules[dst[m]].modules do
          if model.modules[dst[m]].modules[mm].weight then
            model.modules[dst[m]].modules[mm].weight:copy(pre_model.modules[src[m]].modules[mm].weight)
            model.modules[dst[m]].modules[mm].bias:copy(pre_model.modules[src[m]].modules[mm].bias)
            model.modules[dst[m]].modules[mm].gradWeight:copy(pre_model.modules[src[m]].modules[mm].gradWeight)
            model.modules[dst[m]].modules[mm].gradBias:copy(pre_model.modules[src[m]].modules[mm].gradBias)
          end
        end
      end
    end

    print("New model ".. #model.modules)
    print("Old model ".. #pre_model.modules)

    --    pre_model = nil
    --    collectgarbage()
  else
    paths.dofile('models/CostVolMulti.lua')

    local pre_model = loadDataParallel(opt.retrain, opt.nGPU) -- defined in util.lua

    -- if there is a model, there has to be an optimState
    assert(paths.filep(opt.optimState), 'File not found: ' .. opt.optimState)

    if opt.nGPU>0 then
      pre_model:cuda()
    end

    model = pre_model
  end
else
  paths.dofile('models/' .. opt.netType .. '.lua')
  print('=> Creating model from file: models/' .. opt.netType .. '.lua')
  model = createModel(opt) -- for the model creation code, check the models/ folder

  if opt.backend == 'cudnn' then
    require 'cudnn'
    cudnn.convert(model, cudnn)
  elseif opt.backend ~= 'nn' then
    error'Unsupported backend'
  end
end

-- 2. Create Criterion
criterion = nn.L2Criterion()
occ_criterion = nn.L2Criterion()

-- Photometric loss
if opt.pme_criterion == 'BCC' then
  print('Using MBCC for pme')
  pme_criterion = nn.MBCCriterion()
elseif opt.pme_criterion == 'SSIM' then
  print('Using MSSIM for pme')
  pme_criterion = nn.MSSIML1Criterion()
  pme_criterion.alpha = 1
elseif opt.pme_criterion == 'SSIML1' then
  print('Using MSSIM&L1 for pme')
  pme_criterion = nn.MSSIML1Criterion()
  pme_criterion.alpha = 0.85
elseif opt.pme_criterion == 'CSAD' then
  print('Using MCSAD for pme')
  pme_criterion = nn.MCSADCriterion()
elseif opt.pme_criterion == 'OBCC' then
  print('Using OBCC for pme')
  pme_criterion = nn.OBCCriterion()
elseif opt.pme_criterion == 'OBGCC' then
  print('Using OBGCC for pme')
  pme_criterion = nn.OBGCCriterion()
  pme_criterion.alpha = opt.pme_alpha
  pme_criterion.beta = opt.pme_beta
  pme_criterion.gamm = opt.pme_gamma
elseif opt.pme_criterion == 'OSSIM' then
  print('Using OSSIM for pme')
  pme_criterion = nn.OSSIML1Criterion()
  pme_criterion.alpha = 1
elseif opt.pme_criterion == 'OSSIML1' then
  print('Using OSSIM&L1 for pme')
  pme_criterion = nn.OSSIML1Criterion()
  pme_criterion.alpha = 0.85
elseif opt.pme_criterion == 'OCSAD' then
  print('Using OCSAD for pme')
  pme_criterion = nn.OCSADCriterion()
end

if pme_criterion then
  pme_criterion.F = opt.frames
  pme_criterion.past_flow = opt.past_flow

  if opt.pme_penalty == 'L1' then
    pme_criterion.p = L1Penalty()
  elseif opt.pme_penalty == 'Lorentzian' then
    pme_criterion.p = LorentzianPenalty()
  end
end

if opt.dataset == 'Kitti2015' then
  pme_criterion.p = L1Penalty(0.38)
end

-- smoothness loss
if opt.smooth_second_order then
  fs_criterion = nn.SecondOrderSmoothnessCriterion()
else
  fs_criterion = nn.SmoothnessCriterion()
end
if opt.smooth_flow_penalty == 'L1' then
  fs_criterion.p = L1Penalty()
elseif opt.smooth_flow_penalty == 'Lorentzian' then
  fs_criterion.p = LorentzianPenalty()
end

-- constant velocity loss
cv_criterion = nn.ConstVelCriterion()

-- occlusion smoothness
os_criterion = nn.SmoothnessCriterion()
if opt.smooth_occ_penalty == 'L1' then
  os_criterion.p = L1Penalty()
elseif opt.smooth_occ_penalty == 'Lorentzian' then
  os_criterion.p = LorentzianPenalty()
elseif opt.smooth_occ_penalty == 'Dirac' then
  os_criterion.p = LorentzianPenalty()
  os_criterion.p:set_eps(0.001)
elseif opt.smooth_occ_penalty == 'KL' then
  os_criterion = nn.KLDivergenceCriterion()
end

-- occlusion prior
oprior_criterion = nn.OcclusionPriorCriterion()

print('=> Model')
print(model)

-- draw graph (the forward graph, '.fg')
--graph.dot(model.fg, 'Forward Graph', paths.concat(opt.save, 'fwd_graph'))
--graph.dot(model.bg, 'Backward Graph', paths.concat(opt.save, 'bwd_graph'))

print('=> Criterion')

if criterion     then print(criterion);     criterion:cuda()     end
if occ_criterion then print(occ_criterion); occ_criterion:cuda() end
if pme_criterion then print(pme_criterion); pme_criterion:cuda() end
if fs_criterion  then print(fs_criterion); fs_criterion:cuda()  end
if os_criterion  then print(os_criterion); os_criterion:cuda()  end
if oprior_criterion  then print(oprior_criterion); oprior_criterion:cuda()  end
if fprior_criterion  then print(fprior_criterion); fprior_criterion:cuda()  end
if mask_criterion then print(mask_criterion); mask_criterion:cuda() end

if opt.sizeAverage == false then
  if criterion     then criterion.sizeAverage = false     end
  if occ_criterion then occ_criterion.sizeAverage = false  end
  if pme_criterion then pme_criterion.sizeAverage = false  end
  if fs_criterion  then fs_criterion.sizeAverage = false   end
  if os_criterion  then os_criterion.sizeAverage = false   end
  if oprior_criterion  then oprior_criterion.sizeAverage = false   end
  if fprior_criterion  then fprior_criterion.sizeAverage = false   end
  if mask_criterion then mask_criterion.sizeAverage = false  end
end

collectgarbage()