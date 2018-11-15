----------------------------------------------------
---- FRIST ORDER SMOOTHNESS CRITERION
-----------------------------------------------------
-- Enforces a contrast-sensitive first order smoothness constraint
--
-- input -> flow field (batchSize x 2 x Height x Width)
-- target -> reference frame (batchSize x ChannelSize x Height x Width)
--
-- Copyright 2018 Joel Janai, Fatma Güney, Anurag Ranjan and the Max Planck Gesellschaft.
-- All rights reserved.
-- This software is provided for research purposes only.
-- By using this software you agree to the terms of the license file
-- in the root folder.
---------------------------------------------------------------

require 'criterions.penalty.quadratic_function'

local SmoothnessCriterion, parent = torch.class('nn.SmoothnessCriterion', 'nn.Criterion')

function SmoothnessCriterion:__init()
	parent.__init(self)
	self.sizeAverage = true       -- normalize by number of pixels if true
  self.gradCheck = false        -- gradient test
  self.p = QuadraticPenalty()   -- penalty function
  self.cs = 20                  -- sensitivity
end

function SmoothnessCriterion:updateOutput(input, target)
    assert( input:size(3) == target:size(3) and  input:size(4) == target:size(4),
      "input and target size mismatch")

    self.buffer = self.buffer or input.new()

    local buffer = self.buffer
    local norm = 1.0 / input:nElement();
    
    -- first order derivative of flow field in x and y direction
    self.gy = torch.Tensor(input:size()):zero()
    self.gx = torch.Tensor(input:size()):zero()
    if(self.gradCheck == false) then
      self.gy = self.gy:cuda()
      self.gx = self.gx:cuda()
    end 

    self.gy[{{},{},{1,input:size(3)-1},{}}]:add(input[{{},{},{2,input:size(3)},{}}], -1, input[{{},{},{1,input:size(3)-1},{}}])
    self.gx[{{},{},{},{1,input:size(4)-1}}]:add(input[{{},{},{},{2,input:size(4)}}], -1, input[{{},{},{},{1,input:size(4)-1}}])
    
    -- forward difference of flow field in x and y direction
    local igy = torch.Tensor(input:size()):zero()
    local igx = torch.Tensor(input:size()):zero()
    if(self.gradCheck == false) then
      igy = igy:cuda()
      igx = igx:cuda()
    end 
    igy[{{},{},{1,target:size(3)-1},{}}]:add(target[{{},{},{2,target:size(3)},{}}], -1, target[{{},{},{1,target:size(3)-1},{}}])
    igx[{{},{},{},{1,target:size(4)-1}}]:add(target[{{},{},{},{2,target:size(4)}}], -1, target[{{},{},{},{1,target:size(4)-1}}])
    
    self.wy = torch.expandAs(torch.exp(-self.cs * torch.mean(torch.abs(igy), 2)), self.gy)
    self.wx = torch.expandAs(torch.exp(-self.cs * torch.mean(torch.abs(igx), 2)), self.gx)
    
    -- apply robust function on second derivative and weight by img gradients
    buffer:resizeAs(input)
    buffer:add(self.p:apply(self.gx):cmul(self.wx), self.p:apply(self.gy):cmul(self.wy))
    buffer = buffer:sum()
    
    if self.sizeAverage then
      self.output = norm * buffer
    else
      self.output = buffer
    end
    
    return self.output    
end

function SmoothnessCriterion:updateGradInput(input, target)
  	assert( input:size(3) == target:size(3) and  input:size(4) == target:size(4),
      "input and target size mismatch")

    self.buffer = self.buffer or input.new()

    local buffer = self.buffer
    local gradInput = self.gradInput
    local norm = 1. / input:nElement()

    self.gy = self.p:der(self.gy):cmul(self.wy)
    self.gx = self.p:der(self.gx):cmul(self.wx)
    
    -- gradients
    local gys1 = torch.Tensor(input:size()):zero()
    local gxs1 = torch.Tensor(input:size()):zero()
    if(self.gradCheck == false) then
      gys1 = gys1:cuda()
      gxs1 = gxs1:cuda()
    end 
    gys1[{{},{},{2,input:size(3)},{}}]:copy(self.gy[{{},{},{1,input:size(3)-1},{}}])
    gxs1[{{},{},{},{2,input:size(4)}}]:copy(self.gx[{{},{},{},{1,input:size(4)-1}}])

    gradInput:resizeAs(input)
    if self.sizeAverage then
      gradInput = norm * (-self.gx  + gxs1 - self.gy + gys1)
    else
      gradInput = (-self.gx + gxs1 -self.gy + gys1)
    end

    return gradInput
end

function SmoothnessCriterion:clear() 
  self.buffer = nil
  self.gy = nil
  self.gx = nil
  self.wy = nil
  self.wx = nil
end


