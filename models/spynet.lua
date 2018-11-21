-- Copyright 2018 Joel Janai, Fatma Güney, Anurag Ranjan and the Max Planck Gesellschaft.
-- All rights reserved.
-- This software is provided for research purposes only.  
-- By using this software you agree to the terms of the license file 
-- in the root folder.
-- For commercial use, please contact ps-license@tue.mpg.de.

require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'os'
require 'stn'
require 'spy'

function volcon_level(opt, input, channels, lvl, ups_flow)
  local SPyNet = input - nn.SpatialConvolution(channels,32,7,7,1,1,3,3) - nn.ReLU()
              - nn.SpatialConvolution(32,64,7,7,1,1,3,3) - nn.ReLU()
              - nn.SpatialConvolution(64,32,7,7,1,1,3,3) - nn.ReLU()
              - nn.SpatialConvolution(32,16,7,7,1,1,3,3) - nn.ReLU()
  
  local ref = 1
    
  local occ
  if opt.frames > 2 then 
    ref = 0.5 * (opt.frames + 1)
    occ = SPyNet - nn.SpatialConvolution(16,2,7,7,1,1,3,3) - nn.SpatialSoftMax(true) 
  end
  
  local flow = SPyNet - nn.SpatialConvolution(16,2,7,7,1,1,3,3)
  -- add upsampled flow to residual flow
  if ups_flow and opt.residual == 1 then
    flow = {flow, ups_flow} - nn.CAddTable()
  end
 
  local warping = {}
  for f = 1,opt.frames do
    if f ~= ref then
      local a = (f - 1) * 3 + 1
  
      local imgOut = input 
                      - nn.Narrow(2,a,3) 
                      - nn.Transpose({2,3},{3,4})
                     
      local floOut
      if opt.rescale_flow == 1 then
        floOut = flow - nn.MulConstant(opt.flownet_factor * (f - ref)) - nn.Transpose({2,3},{3,4})
      else
        floOut = flow - nn.MulConstant(opt.flownet_factor * (f - ref) / math.pow(2, lvl)) - nn.Transpose({2,3},{3,4})
      end
      
      table.insert(warping, {imgOut, floOut} 
                      - nn.BilinearSamplerBHWD() 
                      - nn.Transpose({3,4},{2,3}))
    end
  end

  local output = {flow, unpack(warping)}
  if opt.frames > 2 then
    output = {flow, occ, unpack(warping)}
  end
   
   return output
end

function createModel(opt)
  local nGPU = opt.nGPU
--  nngraph.setDebug(true)

  local out_warp_start, n_unit_out
  local ref = 1
  if opt.frames == 2 then
    out_warp_start = 2
    n_unit_out = 2 -- only flow and warped
  else
    -- idx to ref
    ref = 0.5 * (opt.frames + 1)
    out_warp_start = 3
    n_unit_out = opt.frames + 1 -- flow + occ + warped
  end
  local ref_c = (ref - 1) * 3 + 1
  
  -- scale pyramid
  local inputData = -nn.Identity()
  local downs_input = {}
  downs_input[opt.levels] = inputData
  for l = (opt.levels-1),1,-1 do
    downs_input[l] = downs_input[l+1] - nn.SpatialAveragePooling(2,2,2,2)
  end

  -- each level
  local out_level = {}
  local flow_scale = {}
  local output = {}
  for l = 1, opt.levels do    
    if l == 1 then
      out_level[l] = volcon_level(opt, downs_input[l], opt.channels, opt.levels - l)
    else
      local ups_flow = out_level[l-1][1] - nn.SpatialUpSamplingBilinear(2.0)
      if opt.rescale_flow == 1 then 
        ups_flow = ups_flow - nn.MulConstant(2)
      end
      
      local input_table = {}
      for f = 1,opt.frames do
        local a = (f - 1) * 3 + 1
        
        if f ~= ref then
          local imgOut = downs_input[l] 
                          - nn.Narrow(2,a,3) 
                          - nn.Transpose({2,3},{3,4})
                         
          local floOut
          if opt.rescale_flow == 1 then
            floOut = ups_flow - nn.MulConstant(opt.flownet_factor * (f - ref)) - nn.Transpose({2,3},{3,4})
          else
            floOut = ups_flow - nn.MulConstant(opt.flownet_factor * (f - ref) / math.pow(2, opt.levels - l)) - nn.Transpose({2,3},{3,4})
          end
          
          table.insert(input_table, {imgOut, floOut} 
                          - nn.BilinearSamplerBHWD() 
                          - nn.Transpose({3,4},{2,3}))      
        else
          table.insert(input_table, downs_input[l] - nn.Narrow(2,a,3))        
        end
      end
      
      -- add residual flow to input
      local chs = opt.channels
      if opt.flow_input == 1 then
        table.insert(input_table, ups_flow)
        chs = chs + 2
      end
      if opt.frames > 2 and opt.occ_input == 1 then
        local ups_occ = out_level[l-1][2] - nn.SpatialUpSamplingNearest(2.0)
        table.insert(input_table, ups_occ)
        chs = chs + 2
      end
      local input_level = input_table - nn.JoinTable(1,3)
      
      out_level[l] = volcon_level(opt, input_level, chs, opt.levels - l, ups_flow)
      
      -- add residual flow to output flow
      if opt.residual == 1 then
        out_level[l][1] = {out_level[l][1], ups_flow} - nn.CAddTable()
      end
    end
    
    for i = 1, n_unit_out do
      output[(opt.levels - l) * n_unit_out + i] = out_level[l][i]
    end
    
    if opt.rescale_flow == 1 then
      table.insert(flow_scale, opt.flownet_factor)
    else
      table.insert(flow_scale, opt.flownet_factor/math.pow(2,opt.levels - l))
    end
  end
  
  
  local model = nn.gModule({inputData}, output)
  model.flow_scale = flow_scale
  
--  model:parameters()[1][{{},{1,3},{},{}}]:set(0) -- TEST WITHOUT PAST FRAME

   if nGPU>0 then
      model:cuda()
      model = makeDataParallel(model, nGPU)
   end
   
   return model
end