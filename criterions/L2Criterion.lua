----------------------------------------------------
---- END POINT ERROR CRITERION
-----------------------------------------------------
-- Computes average endpoint error between estimated flow and gt flow 
--  for masked pixels.
--
-- input -> estimated flow (batchSize x 2 x Height x Width)
-- target[1] -> gt flow (batchSize x 2 x Height x Width)
-- target[2] -> mask (batchSize x 1 x Height x Width)
--
-- Copyright 2018 Joel Janai, Fatma Güney, Anurag Ranjan and the Max Planck Gesellschaft.
-- All rights reserved.
-- This software is provided for research purposes only.
-- By using this software you agree to the terms of the license file
-- in the root folder.
---------------------------------------------------------------

local L2Criterion, parent = torch.class('nn.L2Criterion', 'nn.Criterion')

local  eps = 1e-12

function L2Criterion:__init()
	parent.__init(self)
	self.sizeAverage = true
end

function L2Criterion:updateOutput(input, target)
	assert( input:nElement() == target[1]:nElement(),
    "input and target size mismatch")

    local output
    local npixels

    npixels = target[2]:sum()   

    local buffer = torch.add(input, -1, target[1]):pow(2)
    self.epe_map = torch.sum(buffer,2):sqrt():cmul(target[2])   -- second channel is flow
    output = self.epe_map:sum()

    if self.sizeAverage then
      output = output / npixels
    end

    return output   
end

function L2Criterion:updateGradInput(input, target)

	assert( input:nElement() == target[1]:nElement(),
    "input and target size mismatch")

    local gradInput = self.gradInput
    local npixels
    local loss

    npixels = target[2]:sum()    

    local buffer = torch.add(input, -1, target[1]):pow(2)
    loss = torch.sum(buffer,2):cmul(target[2]):sqrt():add(eps)  -- forms the denominator
    loss = loss:repeatTensor(1,input:size(2),1,1)   -- Repeat tensor to scale the gradients

    local rep_mask = torch.repeatTensor(target[2],1,input:size(2),1,1)
    gradInput:resizeAs(input)
    gradInput:add(input, -1, target[1]):cdiv(loss):cmul(rep_mask)
    
    if self.sizeAverage then
      gradInput = gradInput / npixels  
    end

    return gradInput
end

function L2Criterion:clear() 
  self.epe_map = nil
end