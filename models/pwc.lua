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

require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'nngraph'

require 'stn'
require 'spy'

paths.dofile('CostVolMulti.lua')
local d = 16

-- test 
function testMulti()
  local bs = 8
  local h = 320
  local w = 640
  local frames = 3
  local levels = 4
  
  local ref = 1
  if frames > 2 then
    ref = 0.5 * (frames + 1)
  end
  
  local input = torch.rand(bs, frames*3, h, w)
  input = input:cuda()
  
  local model = createModelMulti()
  
  model:forward(input)
  local outputTable = model.output

  for i = 1, #outputTable do
    print(outputTable[i]:size())
  end
end

-- convolutions for feature extraction
function convUnit(d_in, d_out, stride)
  local unit = nn.Sequential()
  unit:add(nn.SpatialConvolution(d_in, d_out, 3, 3, stride, stride, 1, 1))
  unit:add(nn.LeakyReLU(0.2, true))
  unit:add(nn.SpatialConvolution(d_out, d_out, 3, 3, 1, 1, 1, 1))
  unit:add(nn.LeakyReLU(0.2, true))
  return unit
end

-- warping using BilinearSampler
function warpingUnit(I, F)
  local input = I - nn.Transpose({2,3}, {3,4})       
  local flow = F - nn.Transpose({2,3}, {3,4})
  local W = {input, flow} - nn.BilinearSamplerBHWD() - nn.Transpose({3,4}, {2,3})
  return W
end

-- flow and occlusion decoder
function decoder(nChannels)
  local m = nn.Sequential()
  m:add(nn.SpatialConvolution(nChannels, d*8, 3, 3, 1, 1, 1, 1)):add(nn.LeakyReLU(0.2, true))
  m:add(nn.SpatialConvolution(d*8, d*8, 3, 3, 1, 1, 1, 1)):add(nn.LeakyReLU(0.2, true))
  m:add(nn.SpatialConvolution(d*8, d*6, 3, 3, 1, 1, 1, 1)):add(nn.LeakyReLU(0.2, true))
  m:add(nn.SpatialConvolution(d*6, d*4, 3, 3, 1, 1, 1, 1)):add(nn.LeakyReLU(0.2, true))
  m:add(nn.SpatialConvolution(d*4, d*2, 3, 3, 1, 1, 1, 1)):add(nn.LeakyReLU(0.2, true))
  m:add(nn.SpatialConvolution(d*2, 2, 3, 3, 1, 1, 1, 1))
  return m
end

function createModelMulti(opt)
  local win, frames, levels, nGPU = 5, 3, 4, 1
  local featMaps = {3, d, d*2, d*4, d*6, d*8, d*12} 
  
  local two_frame = 0           -- standard pwc
  local pwc_sum_cvs = false     -- sum future and past cost volumes
  local skip = 2                -- skip levels
  local flownet_factor = 20     -- flow scaling factor
  local rescale_flow = 0        -- rescale flow while downsampling
  
  local pwc_res = 0             -- residual 
  local occ_input = 0           -- input previous occlusion  
  local siamese = 1             -- use siamese network (1) or image (0)
  
  local past_flow = false   -- add past flow decoder (true) or assume constant velocity (false)
  
  if opt then
    win = opt.pwc_ws
    frames = opt.frames
    levels = opt.levels
    occ_input = opt.occ_input
    nGPU = opt.nGPU
    pwc_res = opt.residual
    skip = opt.pwc_skip
    siamese = opt.pwc_siamese
    two_frame = opt.two_frame
    flownet_factor = opt.flownet_factor
    rescale_flow = opt.rescale_flow
    pwc_sum_cvs = opt.pwc_sum_cvs
    past_flow = opt.past_flow
  end
  

  if skip == 0 then
    featMaps[1] = featMaps[2]
  end

  -- use image instead of siamese network
  if siamese == 0 then
    featMaps = {3, 3, 3, 3, 3, 3, 3, 3, 3}
  end

  -- reference frame
  local ref = 1
  if frames > 2 then
    ref = 0.5 * (frames + 1)
  end
  
  -- get finest level
  local l_st = math.max(skip + 1, 1)
  
  -- extract frames from input
  local inputData = - nn.Identity()
  local Is = {}
  for f = 1, frames do
    local a = (f - 1) * 3 + 1
    local I = inputData - nn.Narrow(2,a,3) 
    table.insert(Is, I)
  end
  
  -- warping output
  local ds = {}
  for f = 1, frames do
    if f ~= ref then
      ds[f] = {} 
      ds[f][1] = Is[f]

      for l = 2, levels - l_st + 1 do 
        ds[f][l] = ds[f][l-1] - nn.SpatialAveragePooling(2,2,2,2) 
      end
    end
  end


  local f_i = 1
  local l_i = frames
  if two_frame == 1 then
    f_i = ref
    l_i = ref + 1
  end
  
  -- create siamese network
  local feats = {}
  feats[f_i] = {}
  if skip == 0 then
    if siamese == 1 then
      feats[f_i][1] = convUnit(3, featMaps[1], 1)
    else
      feats[f_i][1] = nn.Identity()
    end
  end
  for l = 2, levels do 
    if siamese == 1 then
      feats[f_i][l] = convUnit(featMaps[l-1], featMaps[l], 2)
    else 
      feats[f_i][l] = nn.SpatialAveragePooling(2,2,2,2)
    end
  end

  -- clone it frames-1 times
  for f = f_i+1, l_i do
    feats[f] = {}
    if skip == 0 then
      feats[f][1] = feats[f_i][1]:clone('weight', 'bias', 'gradWeight', 'gradBias')
    end
    for l = 2, levels do
      feats[f][l] = feats[f_i][l]:clone('weight', 'bias', 'gradWeight', 'gradBias')
    end
  end

  local cs = {}
  for f = f_i, l_i do
    cs[f] = {}
    for l = 1, levels do
      if l == 1 then
        if skip == 0 then
          cs[f][l] = Is[f] - feats[f][l]
        else
          cs[f][l] = Is[f]
        end
      else
        cs[f][l] = cs[f][l-1] - feats[f][l]
      end
    end
  end
  
  local cvs_fwd = {}
  local cvs_bwd = {}
  local fs = {}
  local bfs = {}
  local ufs = {}
  local ubfs = {}
  local skip_ufs = {}
  local skip_ubfs = {}
  local occs = {}
  local uoccs = {}
  local skip_occs = {}
  local ws = {}
  local iws = {}
  
  -- table of warped features and warped images
  for f = 1, frames do
    ws[f] = {}
    iws[f] = {}
  end
  
  local input
    
  -- create each level of pwc net
  local flow_scale = {}
  for l = levels, l_st, -1 do
      if l == levels then
        -- use extracted features on coarsest level
        input = cs
      else
        -- use warped features otherwise
        input = ws
      end
  
      -- future features
      local future = {cs[ref][l]}
      for f = ref+1, l_i do
        future[f-ref+1] = input[f][l]
      end
      -- future cost volume
      cvs_fwd[l] = future - nn.CostVolMulti(win, true)
      
      local cvs_flow, cvs_occ, nd_flow, nd_occ
      local nd = win*win
      if frames > 2 and two_frame == 0 then
        -- past features
        local past = {cs[ref][l]}
        for f = ref-1, 1, -1 do
          past[ref-f+1] = input[f][l]
        end
        -- past cost volume
        cvs_bwd[l] = past - nn.CostVolMulti(win, false)
        
        -- join or sum cost volumes 
        if pwc_sum_cvs == false then
          cvs_flow = {cvs_fwd[l], cvs_bwd[l]} - nn.JoinTable(2)
          cvs_occ = cvs_flow
          nd_flow = nd * 2
          nd_occ = nd * 2
        else
          cvs_flow = {cvs_fwd[l], cvs_bwd[l]} - nn.CAddTable(2)
          cvs_occ = {cvs_fwd[l], cvs_bwd[l]} - nn.JoinTable(2)
          nd_flow = nd 
          nd_occ = nd * 2
        end
      else
        -- use just future in two frame case
        cvs_flow = cvs_fwd[l]
        cvs_occ = cvs_flow
        nd_flow = nd 
        nd_occ = nd
      end
          
      -- occlusion decoder if more than 2 frames
      if frames > 2 then 
        -- input to occlusion decoder: joint cost volume and reference features
        local decoderO_in = {cvs_occ, cs[ref][l]}
        local decoder_O_inN = nd_occ + featMaps[l]
        if two_frame == 1 then
          -- in two frame setup add target frame
          table.insert(decoderO_in, cs[ref+1][l])
          decoder_O_inN = decoder_O_inN + featMaps[l]
        end
        
        -- add last level flow
        if l ~= levels then
          table.insert(decoderO_in, ufs[l+1])
          decoder_O_inN = decoder_O_inN + 2
          if occ_input == 1 then
            -- add last level occlusions
            table.insert(decoderO_in, uoccs[l+1])
            decoder_O_inN = decoder_O_inN + 2
          end
        end
        
        -- occlusion decoder with spatial softmax
        occs[l] = decoderO_in - nn.JoinTable(2) - decoder(decoder_O_inN) - nn.SpatialSoftMax(true)
        
        -- upsample for next level input
        if skip > 0 or occ_input == 1 then
          uoccs[l] = occs[l] - nn.SpatialUpSamplingNearest(2.0)
        end
        
        -- upsamples to output resolution
        if skip > 0 then
          skip_occs[l] = uoccs[l]
          for i = 2, l_st-1 do
            skip_occs[l] = skip_occs[l] - nn.SpatialUpSamplingNearest(2.0)
          end
        end
      end
      
      -- flow decoders
      if l == levels then
        -- coarsest level: only cost volume 
        fs[l] = cvs_flow - decoder(nd_flow) 
        if past_flow then
          -- past flow decoder
          bfs[l] = cvs_flow - decoder(nd_flow)
        end
      else
        -- all other levels: cost volume, last future flow, (last past flow)
        local decoderF = {cvs_flow, cs[ref][l], ufs[l+1]} - nn.JoinTable(2) - decoder(nd_flow + featMaps[l] + 2)
        local decoderBF
        if past_flow then
          decoderBF = {cvs_flow, cs[ref][l], ubfs[l+1]} - nn.JoinTable(2) - decoder(nd_flow + featMaps[l] + 2)
        end
        
        -- residual flow add upsampled flow from last layer after decoder
        if pwc_res == 1 then -- residual flow
          fs[l] = {decoderF, ufs[l+1]} - nn.CAddTable()
          if past_flow then
            bfs[l] = {decoderBF, ubfs[l+1]} - nn.CAddTable()
          end
        else
          fs[l] = decoderF
          if past_flow then
            bfs[l] = decoderBF
          end
        end
      end
          
      -- upsample flow and scale
      -- l_st = 2  
      --    -> ufs l_1  -> skip_ufs l_1
      -- l_st = 3  
      --    -> ufs l_2  -> skip_ufs l_1
      if skip > 0 or l > l_st then
        ufs[l] = fs[l] - nn.SpatialUpSamplingBilinear(2.0)
        if past_flow then
          ubfs[l] = bfs[l] - nn.SpatialUpSamplingBilinear(2.0)
        end
        if rescale_flow == 1 then
          ufs[l] = ufs[l] - nn.MulConstant(2.0)
          if past_flow then
            ubfs[l] = ubfs[l] - nn.MulConstant(2.0)
          end
        end
        
        -- upsamples to output resolution
        if skip > 0 then
          skip_ufs[l] = ufs[l]
          if past_flow then
            skip_ubfs[l] = ubfs[l]
          end
          for i = 2, l_st-1 do
            skip_ufs[l] = skip_ufs[l] - nn.SpatialUpSamplingBilinear(2.0)
            if past_flow then
              skip_ubfs[l] = skip_ubfs[l] - nn.SpatialUpSamplingBilinear(2.0)
            end
            if rescale_flow == 1 then
              skip_ufs[l] = skip_ufs[l] - nn.MulConstant(2.0)
              if past_flow then
                skip_ubfs[l] = skip_ubfs[l] - nn.MulConstant(2.0)
              end
            end
          end
        end
      end
          
      -- warp features and images according to flow from previous levels
      for f = 1, frames do
        if f ~= ref then                
          if l > l_st and f >= f_i and f <= l_i then
            -- independent of l_st
            -- l = 2, (l-1) = 1, s = 1/2^0
            -- l = 3, (l-1) = 2, s = 1/2^1
            -- l = 4, (l-1) = 3, s = 1/2^2
            local ufm
            if rescale_flow == 1 then
              ufm = ufs[l] - nn.MulConstant(flownet_factor * (f - ref))
            else
              ufm = ufs[l] - nn.MulConstant(flownet_factor * (f - ref)/math.pow(2,l-2)) 
            end
  
            -- warp the higher res feats with upsampled flow
            ws[f][l-1] = warpingUnit(cs[f][l-1], ufm)
          end
          
          -- upsampled flow multiplied
          -- dependent of l_st
          -- l_st = 2
          --    l = 2, (l-2) = 0, s = 1/2^0
          --    l = 3, (l-2) = 1, s = 1/2^1
          --    l = 4, (l-2) = 2, s = 1/2^2
          -- l_st = 3
          --    
          --    l = 3, (l-3) = 0, s = 1/2^0
          --    l = 4, (l-3) = 1, s = 1/2^1
          
          -- warp the image of higher res with upsampled flow
          local tmp_ufm
          if skip == 0 then
            if past_flow and f < ref then
              tmp_ufm = bfs[l]
            else
              tmp_ufm = fs[l]
            end
          else
            if past_flow and f < ref then
              tmp_ufm = skip_ubfs[l]
            else
              tmp_ufm = skip_ufs[l]
            end
          end
          
          -- NOTE: past is left negative to copy weights of pretrained model with only future flow
          local skip_ufm
          if rescale_flow == 1 then
            skip_ufm = tmp_ufm - nn.MulConstant(flownet_factor * (f - ref))
          else
            skip_ufm = tmp_ufm - nn.MulConstant(flownet_factor * (f - ref)/math.pow(2,l-l_st))
          end
          
          iws[f][l] = warpingUnit(ds[f][l-l_st+1], skip_ufm)
        end -- end of if ref
      end -- end of frames
      
      -- rescaling flow
      if rescale_flow == 1 then
        table.insert(flow_scale, flownet_factor)
      else
        table.insert(flow_scale, flownet_factor/math.pow(2,l-l_st))
      end
  end
    
  -- output future flow, (past flow), occlusions, warped images
  local outputTable = {}
  for l = l_st, levels do
    if skip == 0 then
      -- future flow, (past flow)
      table.insert(outputTable, fs[l])
      if past_flow then
        table.insert(outputTable, bfs[l])
      end
      -- occlusions
      if frames > 2 then
        table.insert(outputTable, occs[l])
      end
    else
      -- future flow, (past flow)
      table.insert(outputTable, skip_ufs[l])
      if past_flow then
        table.insert(outputTable, skip_ubfs[l])
      end
      -- occlusions
      if frames > 2 then
        table.insert(outputTable, skip_occs[l])
      end
    end

    -- warped images
    for f = 1, frames do
      if f ~= ref then
        table.insert(outputTable, iws[f][l])
      end
    end
  end
  
  -- create model
  local model = nn.gModule({inputData}, outputTable)
  model.flow_scale = flow_scale
  model.past_flow = past_flow
  
  -- visualize model
--  graph.dot(model.fg, 'multi model fwd', 'pcw multi fwd')
--  graph.dot(model.bg, 'model bwd', 'pcw bwd')

  -- cuda
  if nGPU > 0 then
    model:cuda()
    
    model = makeDataParallel(model, nGPU)
  end
   
  return model
end

createModel = createModelMulti
--
--paths.dofile('../util.lua')
--testMulti()
