----------------------------------------------------
---- COST VOLUME MODULE
-----------------------------------------------------
-- Computes a cost volume given F input feature maps (with respect to the first feature map)
--
-- input -> table consisting of the {feature map 1 (reference), ... , feature map F}
--  feature map f -> (batchSize x ChannelSize x Height x Width)
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

local CostVolMulti, parent = torch.class('nn.CostVolMulti', 'nn.Module')

function CostVolMulti:__init(win, fwd, verbose)
  parent.__init(self)
  -- window
  if win then
    self.win = win
  else
    self.win = 3
  end
  
  -- mirror window in case of past frames 
  if fwd ~= nil then
    self.fwd = fwd
  else
    self.fwd = true
  end

  -- verbosity
  if verbose then
    self.verbose = verbose
  else
    self.verbose = false
  end
  
  self.gradInput = {torch.Tensor(), torch.Tensor()}
end

function CostVolMulti:updateOutput(input)
  local frames = #input
  for f = 2, frames do
    assert(input[f]:nElement() == input[f-1]:nElement(), "input sizes mismatch")
  end
  
  local ref = input[1]
  local N, h, w = ref:size(2), ref:size(3), ref:size(4)
  local n = 0.5 * (self.win - 1)
  
  self.output:resize(ref:size(1), self.win * self.win, h, w):zero()
    
  -- shift according to window  
  for f = 2, frames do
    local frame = input[f]

    local i = 1
    for q_x_ = -n, n do
      for q_y_ = -n, n do
        local q_x = q_x_ * (f-1)
        local q_y = q_y_ * (f-1) 
        
        if self.fwd == false then
          q_x = q_x * -1
          q_y = q_y * -1
        end
             
        local qx = {1 + q_x, w}
        local px = {1, w - q_x}
        if (q_x < 0) then
          qx = {1, w + q_x}
          px = {1 - q_x, w}
        end
        local qy = {1 + q_y, h}
        local py = {1, h - q_y}
        if (q_y < 0) then
          qy = {1, h + q_y}
          py = {1 - q_y, h}
        end
        
        local cost = torch.cmul(ref[{{},{},qy,qx}], frame[{{},{},py,px}])
        self.output[{{},i,qy,qx}]:add(cost:sum(2))
        
        i = i + 1
      end -- end of q_y
    end -- end of q_x
    
    -- collect garbage otherwise excessive storage usage with multi GPU
    collectgarbage()
  end -- end of frame
  
  self.output:div(N * (frames - 1))
  
  if self.verbose then
    print(#self.output)
    print(self.output[{{1},{},50,50}]:reshape(self.win,self.win))
    print(self.output[{{1},{},25,25}]:reshape(self.win,self.win))
  end

  return self.output
end

function CostVolMulti:updateGradInput(input, gradOutput)
  local frames = #input
  
  local ref = input[1]
  local bs, N, h, w = ref:size(1), ref:size(2), ref:size(3), ref:size(4)
  local n = 0.5 * (self.win - 1)

  if #self.gradInput ~= frames then
    for f = 1, frames do
      self.gradInput[f] = self.gradInput[f] or input[f].new()
    end
  end
  
  for f = 1, frames do
    self.gradInput[f]:resizeAs(input[f]):zero()
  end
  
  local gradInputRef = self.gradInput[1]
  
  -- shift according to window
  for f = 2, frames do
    local frame = input[f]
    local gradInputFrame = self.gradInput[f]
    
    local i = 1
    for q_x_ = -n, n do
      for q_y_ = -n, n do
        local q_x = q_x_ * (f-1)
        local q_y = q_y_ * (f-1) 
        
        if self.fwd == false then
          q_x = q_x * -1
          q_y = q_y * -1
        end
                
        local qx = {1 + q_x, w}
        local px = {1, w - q_x}
        if (q_x < 0) then
          qx = {1, w + q_x}
          px = {1 - q_x, w}
        end
        local qy = {1 + q_y, h}
        local py = {1, h - q_y}
        if (q_y < 0) then
          qy = {1, h + q_y}
          py = {1 - q_y, h}
        end
      
        local ny = qy[2]-qy[1]+1
        local nx = qx[2]-qx[1]+1
        local go = gradOutput[{{},i,qy,qx}]:clone():view(bs,1,ny,nx)
        go = torch.repeatTensor(go, 1, N, 1, 1)
        
        gradInputRef[{{},{},qy,qx}]:add(torch.cmul(go, frame[{{},{},py,px}]))   
        gradInputFrame[{{},{},py,px}]:add(torch.cmul(go, ref[{{},{},qy,qx}])) 
        go = nil    
        
        i = i + 1
      end -- end of q_y
    end -- end of q_x
    
    -- collect garbage otherwise excessive storage usage with multi GPU
    collectgarbage()
  end -- end of frame
  
  for f = 1, frames do
    self.gradInput[f]:div(N * (frames - 1))
  end
  
  return self.gradInput
end

function CostVolMulti:clearState()
   return parent.clearState(self)
end

function CostVolMulti:__tostring__()
  return torch.type(self) ..
      string.format('window size = %d', self.win) 
end

--function testJacobian()
--  -- parameters
--  local precision = 1e-5
--  local jac = nn.ModifiedJacobian
  
--  local ws = 5
--  local bs = 1
--  local h = 32
--  local w = 64
--  local c = 2
--  local f = 7

--  -- define inputs and module
--  local input = {}
--  for i = 1, f do
--    input[i] = torch.rand(bs, c, h, w)
--  end
  
--  local module = nn.CostVolMulti(ws)

--  -- test backprop, with Jacobian
--  local err = jac.testJacobianTable(module, input)
--  print('==> error: ' .. err)
--  if err < precision then
--    print('==> module OK')
--  else
--    print('==> error too large, incorrect implementation')
--  end
--end

--paths.dofile('../ModifiedJacobian.lua')
--testJacobian()

--function test() 
--  local win = 5
--  local frames = 3
--  local ref = 0.5*(frames+1)
--  
--  local future = {}
--  local past = {}
--  
--  local img = torch.Tensor(1,1,6,6):fill(0)
--  img[{{1},{1},{3},{3}}]:fill(1)
--  table.insert(past, img)
--  table.insert(future, img)
--  
--  for i = 1,ref do
--    img = torch.Tensor(1,1,6,6):fill(0)
--    img[{{1},{1},{3+i},{3+i}}]:fill(1)
--    table.insert(future, img)
--    
--    img = torch.Tensor(1,1,6,6):fill(0)
--    img[{{1},{1},{3-i},{3-i}}]:fill(1)
--    table.insert(past, img)
--  end
--  
--  local fwd = nn.CostVolMulti(win, true)
--  local bwd = nn.CostVolMulti(win, false)
--  
--  fwd:forward(future)
--  bwd:forward(past)
--end
--test()
