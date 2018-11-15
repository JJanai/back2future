----------------------------------------------------
---- SECOND ORDER SMOOTHNESS CRITERION
-----------------------------------------------------
-- Enforces a contrast-sensitive second order smoothness constraint
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

local SecondOrderSmoothnessCriterion, parent = torch.class('nn.SecondOrderSmoothnessCriterion', 'nn.Criterion')

function SecondOrderSmoothnessCriterion:__init()
	parent.__init(self)
	self.sizeAverage = true       -- normalize by number of pixels if true
  self.gradCheck = false        -- gradient test
  self.p = QuadraticPenalty()   -- penalty function
  self.cs = 20                  -- sensitivity
end

function SecondOrderSmoothnessCriterion:updateOutput(input, target)
    assert( input:size(3) == target:size(3) and  input:size(4) == target:size(4),
      "input and target size mismatch")

    self.buffer = self.buffer or input.new()

    local buffer = self.buffer
    local norm = 1.0 / input:nElement();
    
    -- second order derivative of flow field in x and y direction
    self.gy = torch.Tensor(input:size()):zero()
    self.gx = torch.Tensor(input:size()):zero()
    if(self.gradCheck == false) then
      self.gy = self.gy:cuda()
      self.gx = self.gx:cuda()
    end 

    self.gy[{{},{},{2,input:size(3)-1},{}}]:add(2*input[{{},{},{2,input:size(3)-1},{}}], -1, input[{{},{},{1,input:size(3)-2},{}}]):add(-1, input[{{},{},{3,input:size(3)},{}}])
    self.gx[{{},{},{},{2,input:size(4)-1}}]:add(2*input[{{},{},{},{2,input:size(4)-1}}], -1, input[{{},{},{},{1,input:size(4)-2}}]):add(-1, input[{{},{},{},{3,input:size(4)}}])
    
    -- forward difference of flow field in x and y direction
    local igy = torch.Tensor(input:size(1),1,input:size(3),input:size(4)):zero()
    local igx = torch.Tensor(input:size(1),1,input:size(3),input:size(4)):zero()
    if(self.gradCheck == false) then
      igy = igy:cuda()
      igx = igx:cuda()
    end 
    igy[{{},{},{2,target:size(3)},{}}]:add(torch.mean(torch.add(target[{{},{},{2,target:size(3)},{}}], -1, target[{{},{},{1,target:size(3)-1},{}}]):abs(),2))
    igx[{{},{},{},{2,target:size(4)}}]:add(torch.mean(torch.add(target[{{},{},{},{2,target:size(4)}}], -1, target[{{},{},{},{1,target:size(4)-1}}]):abs(),2))
    igy[{{},{},{2,target:size(3)-1},{}}]:add(torch.mean(torch.add(target[{{},{},{2,target:size(3)-1},{}}], -1, target[{{},{},{3,target:size(3)},{}}]):abs(),2))
    igx[{{},{},{},{2,target:size(4)-1}}]:add(torch.mean(torch.add(target[{{},{},{},{2,target:size(4)-1}}], -1, target[{{},{},{},{3,target:size(4)}}]):abs(),2))
    
    self.wy = torch.expandAs(torch.exp(-self.cs * igy), self.gy)
    self.wx = torch.expandAs(torch.exp(-self.cs * igx), self.gx)
    
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

function SecondOrderSmoothnessCriterion:updateGradInput(input, target)
  	assert( input:size(3) == target:size(3) and  input:size(4) == target:size(4),
      "input and target size mismatch")

    self.buffer = self.buffer or input.new()

    local buffer = self.buffer
    local gradInput = self.gradInput
    local norm = 1. / input:nElement()

    self.gy = self.p:der(self.gy):cmul(self.wy)
    self.gx = self.p:der(self.gx):cmul(self.wx)
    
    -- gradients
    gradInput:resizeAs(input):zero()
    gradInput[{{},{},{2,input:size(3)-1},{}}]:add(2*self.gy[{{},{},{2,input:size(3)-1},{}}])
    gradInput[{{},{},{},{2,input:size(4)-1}}]:add(2*self.gx[{{},{},{},{2,input:size(4)-1}}])
    gradInput[{{},{},{1,input:size(3)-2},{}}]:add(-1, self.gy[{{},{},{2,input:size(3)-1},{}}])
    gradInput[{{},{},{},{1,input:size(4)-2}}]:add(-1, self.gx[{{},{},{},{2,input:size(4)-1}}])
    gradInput[{{},{},{3,input:size(3)},{}}]:add(-1, self.gy[{{},{},{2,input:size(3)-1},{}}])
    gradInput[{{},{},{},{3,input:size(4)}}]:add(-1, self.gx[{{},{},{},{2,input:size(4)-1}}])

    if self.sizeAverage then
      gradInput:mul(norm)
    end

    return gradInput
end

function SecondOrderSmoothnessCriterion:clear() 
  self.buffer = nil
  self.gy = nil
  self.gx = nil
  self.wy = nil
  self.wx = nil
end


