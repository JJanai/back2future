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

testLogger = optim.myLogger(paths.concat(opt.save, 'test.log'))

local batchNumber
local loss
local avg_epe
local avg_epe_nocc
local avg_epe_occ
local avg_oacc
local avg_occ_acc_bwd
local avg_occ_acc_vis
local avg_occ_acc_fwd
local timer = torch.Timer()

local level_weights = {
  0.005, 0.01, 0.02, 0.08, 0.32, 0.64
}

function test()
   print('==> doing epoch on validation data:')
   print("==> online epoch # " .. epoch)

   batchNumber = 0
   cutorch.synchronize()
   timer:reset()

   -- set the dropouts to evaluate mode
   model:evaluate()

   loss = 0
   avg_epe = 0
   avg_epe_nocc = 0
   avg_epe_occ = 0
   avg_oacc = 0
   avg_occ_acc_bwd = 0
   avg_occ_acc_vis = 0
   avg_occ_acc_fwd = 0
   for i=1,nTest/opt.batchSize do -- nTest is set in 1_data.lua
      local indexStart = (i-1) * opt.batchSize + 1
      local indexEnd = (indexStart + opt.batchSize - 1)
      donkeys:addjob(
         -- work to be done by donkey thread
         function()
            local inputs, labels, masks = testLoader:get(indexStart, indexEnd)
            return inputs, labels, masks
         end,
         -- callback that is run in the main thread once the work is done
         testBatch
      )
   end

   donkeys:synchronize()
   cutorch.synchronize()

   loss = loss / (nTest/opt.batchSize) -- because loss is calculated per batch
   avg_epe = avg_epe / (nTest/opt.batchSize) -- because loss is calculated per batch
   avg_epe_nocc = avg_epe_nocc / (nTest/opt.batchSize) -- because loss is calculated per batch 
   avg_epe_occ = avg_epe_occ / (nTest/opt.batchSize) -- because loss is calculated per batch 
   avg_oacc = avg_oacc / (nTest/opt.batchSize) -- because loss is calculated per batch
   avg_occ_acc_bwd = avg_occ_acc_bwd / (nTest/opt.batchSize) -- because loss is calculated per batch
   avg_occ_acc_vis = avg_occ_acc_vis / (nTest/opt.batchSize) -- because loss is calculated per batch
   avg_occ_acc_fwd = avg_occ_acc_fwd / (nTest/opt.batchSize) -- because loss is calculated per batch
   
   if opt.ground_truth == true then
    testLogger:add{['avg epe (test set)'] = avg_epe,['avg epe non occ (test set)'] = avg_epe_nocc,['avg epe occ (test set)'] = avg_epe_occ,['avg loss (test set)'] = loss,['avg occ acc (test set)'] = avg_oacc,['avg bwd acc (train set)'] = avg_occ_acc_bwd,['avg vis acc (train set)'] = avg_occ_acc_vis,['avg fwd acc (train set)'] = avg_occ_acc_fwd}
     print(string.format('Epoch: [%d][TESTING SUMMARY] Total Time(s): %.2f \t'
                            .. 'average loss (per batch): %.2f \t average epe (per batch): %.2f \t average epe non occ (per batch): %.2f \t average epe occ (per batch): %.2f \t average occ acc (per batch): %.2f (%.2f,%.2f,%.2f)',
                         epoch, timer:time().real, loss, avg_epe, avg_epe_nocc, avg_epe_occ, avg_oacc, avg_occ_acc_bwd, avg_occ_acc_vis, avg_occ_acc_fwd))
   else
    testLogger:add{['avg loss (test set)'] = loss}
     print(string.format('Epoch: [%d][TESTING SUMMARY] Total Time(s): %.2f \t'
                            .. 'average loss (per batch): %.2f \t ',
                         epoch, timer:time().real, loss))
   end

  collectgarbage()

   print('\n')


end -- of test()
-----------------------------------------------------------------------------
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()
local masks = torch.CudaTensor()

function testBatch(inputsCPU, labelsCPU, masksCPU)
  batchNumber = batchNumber + opt.batchSize

  inputs:resize(inputsCPU:size()):copy(inputsCPU)
  labels:resize(labelsCPU:size()):copy(labelsCPU)
  masks:resize(masksCPU:size()):copy(masksCPU)

  local outputs = model:forward(inputs[{{},{1,opt.frames*3},{},{}}]:contiguous())
 
  local ref_c, out_warp_start, n_unit_out, n_flow
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
      
    if opt.backward_flow then
      n_flow = 2
      n_unit_out = n_unit_out + 1
      out_warp_start = 4
    end
  end

  -- DOWNSAMPLE
  local down = nn.SpatialAveragePooling(2,2,2,2):cuda()
  local down_nn =  nn.SpatialAveragePooling(2,2,2,2):cuda()
  local down_sampled = inputs:clone()
  local down_sampled_flow = labels[{{},{1,2},{},{}}]:clone()
  local down_sampled_occ = labels[{{},{3},{},{}}]:clone()

  local levels = #outputs / n_unit_out

    
  local err = 0
  local epe = 0
  local epe_nocc = 0
  local epe_occ = 0
  local oacc = 0
  local occ_acc_bwd = 0
  local occ_acc_fwd = 0
  local occ_acc_vis = 0
  -- Supervised
  if(opt.optimize == 'epe') then
    for l = 0, (levels-1) do
        if l > 0 then
          down_sampled_flow = down:forward(down_sampled_flow)
          if opt.rescale_flow == 1 then
            down_sampled_flow:div(2)
          end
          if opt.frames > 2 then
            down_sampled_occ = down_nn:forward(down_sampled_occ):clone()
          end
        end

        --              1-4, 5-8, 9-12
        local sub_outs = {unpack(outputs, l * n_unit_out + 1, (l+1) * n_unit_out)}
        
        err = err + level_weights[l+1] * opt.epe * criterion:forward(sub_outs[1], down_sampled_flow, masks)
        
        if opt.frames > 2 and (not opt.no_occ) then
          local occ_repeated = down_sampled_occ
          if opt.dis_occ == 1 then
            occ_repeated = torch.repeatTensor(occ_repeated, 1, 2, 1, 1)
            if outputs[out_warp_start-1]:size(2) == 3 then
              occ_repeated[{{},{1},{},{}}]:eq(0)
              occ_repeated[{{},{2},{},{}}]:eq(0.5)
              occ_repeated[{{},{3},{},{}}]:eq(1)
            else
              local tmp1 = occ_repeated[{{},{1},{},{}}]
              local tmp2 = occ_repeated[{{},{2},{},{}}]
              occ_repeated[{{},{1},{},{}}] = torch.eq(tmp1,0):float() + 0.5*torch.eq(tmp1,0.5):float()
              occ_repeated[{{},{2},{},{}}] = torch.eq(tmp2,1):float() + 0.5*torch.eq(tmp2,0.5):float()
              occ_repeated = occ_repeated:cuda()
            end
          end
          
          err = err + level_weights[l+1] * occ_criterion:forward(sub_outs[out_warp_start-1], occ_repeated)
        end
      end
  end
 
  -- compute EPE if provided and (supervised or unsupervised)
  if opt.ground_truth == true then
    -- Flow Supervised Loss    
    local epe_b = criterion:forward(outputs[1], {labels[{{},{1,2},{},{}}], masks})
    if opt.sizeAverage == false then 
--      epe_b = epe_b / (opt.batchSize * opt.fineHeight * opt.fineWidth)
      epe_b = epe_b / masks:sum()
    end
    if opt.flownet_factor ~= 1 then
      epe_b = epe_b * opt.flownet_factor
    end
    epe = epe + epe_b
        
    local lbl_occ = labels[{{},{4},{},{}}]:squeeze():float() -- use 3 occ ground truth
    local occ, epe_map, norm
    -- epe in occluded regions!
    local occ = lbl_occ:ne(0.5):cudaByte()
    local vis_epe_map = criterion.epe_map:clone()
    vis_epe_map = vis_epe_map:maskedFill(occ, 0)
    norm = (1-occ):float():cmul(masksCPU):sum()
    vis_epe_map = vis_epe_map:sum() 
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
    occ_epe_map = occ_epe_map:sum()
    if norm > 0 then
      occ_epe_map = occ_epe_map / norm
      if opt.flownet_factor ~= 1 then
        occ_epe_map = occ_epe_map * opt.flownet_factor
      end
      epe_occ = epe_occ + occ_epe_map
    end
    
    local tmp, occ_est_sharp, occ_map
    local occ_acc = 0
    if opt.frames > 2 then
      if outputs[out_warp_start-1]:size(2) == 1 then
        tmp = outputs[out_warp_start-1]:squeeze():float()
        occ_est_sharp = torch.mul(tmp, 2):round():div(2)
      elseif outputs[out_warp_start-1]:size(2) == 3 then
        _,tmp = torch.max(outputs[out_warp_start-1],2)
        tmp = tmp:float():squeeze()
        occ_est_sharp = torch.div(tmp - 1, 2)
      else
        occ_est_sharp = torch.round((1 - outputs[out_warp_start-1][{{},{1},{},{}}]) + (outputs[out_warp_start-1][{{},{2},{},{}}])):mul(0.5)
        occ_est_sharp = occ_est_sharp:float()
      end
      
      local lbl_occ = labels[{{},{3},{},{}}]:squeeze():float() -- use #frames occ ground truth
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
  criterion:clear()
  
  -- unsupervised
  if(opt.optimize == 'pme') then
    for l = 0,(levels-1) do
      if l > 0 then
        down_sampled = down:forward(down_sampled)
      end
      --              1-4, 5-8, 9-12
      local sub_outs = {unpack(outputs, l * n_unit_out + 1, (l+1) * n_unit_out)}
      
      -- Flow Smoothness Loss
      for i = 1, n_flow do
        err = err + level_weights[l+1] * opt.smooth_flow * fs_criterion:forward(sub_outs[1], down_sampled[{{},{ref_c,ref_c+2},{},{}}])
      end
      fs_criterion:clear()
      
      if opt.backward_flow then
        err = err + level_weights[l+1] * opt.const_vel * cv_criterion:forward(sub_outs)
      end
      
      -- Photometric Loss
      err = err + level_weights[l+1] * opt.pme * pme_criterion:forward(sub_outs, down_sampled[{{},{ref_c,ref_c+2},{},{}}])
      pme_criterion:clear()
      
      -- Occlusion Smoothness Loss
      if opt.frames > 2 and (not opt.no_occ) then
        err = err + level_weights[l+1] * opt.smooth_occ * os_criterion:forward(sub_outs[out_warp_start-1], down_sampled[{{},{ref_c,ref_c+2},{},{}}])
        os_criterion:clear()
        err = err + level_weights[l+1] * opt.prior_occ * oprior_criterion:forward(sub_outs[out_warp_start-1], down_sampled[{{},{ref_c,ref_c+2},{},{}}])
      end
      
      collectgarbage()
    end
	end
   
  cutorch.synchronize()
  local pred = outputs[1]:float()

  loss = loss + err
  avg_epe = avg_epe + epe
  avg_epe_nocc = avg_epe_nocc + epe_nocc
  avg_epe_occ = avg_epe_occ + epe_occ
  avg_oacc = avg_oacc + oacc
  avg_occ_acc_bwd = avg_occ_acc_bwd + occ_acc_bwd
  avg_occ_acc_vis = avg_occ_acc_vis + occ_acc_vis
  avg_occ_acc_fwd = avg_occ_acc_fwd + occ_acc_fwd

  print(('Epoch: Testing [%d][%d/%d]'):format(epoch, batchNumber, nTest))
end
