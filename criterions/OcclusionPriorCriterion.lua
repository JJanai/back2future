----------------------------------------------------
---- OCCLUSIONS PRIOR CRITERION
-----------------------------------------------------
-- Favors the visible state.
--
-- input -> occlusions (batchSize x 2 x Height x Width)
--
-- Copyright 2018 Joel Janai, Fatma Güney, Anurag Ranjan and the Max Planck Gesellschaft.
-- All rights reserved.
-- This software is provided for research purposes only.
-- By using this software you agree to the terms of the license file
-- in the root folder.
---------------------------------------------------------------

require 'criterions.penalty.quadratic_function'

local OcclusionPriorCriterion, parent = torch.class('nn.OcclusionPriorCriterion', 'nn.Criterion')

-- Computes average endpoint error for batchSize x ChannelSize x Height x Width
-- flow fields or general multidimensional matrices.

function OcclusionPriorCriterion:__init()
	parent.__init(self)
	self.sizeAverage = true
  self.penalty = 1
end

function OcclusionPriorCriterion:updateOutput(input, target)
    assert( input:size(3) == target:size(3) and  input:size(4) == target:size(4),
      "input and target size mismatch")
    
    local norm = input:size(2) / input:nElement();

    local output = input.new()
    output:resize(input:size(1),1,input:size(3),input:size(4))
    if input:size(2) == 3 then
      output[{{},{1},{},{}}] = (1 - input[{{},{2},{},{}}]):cmul(torch.add(input[{{},{1},{},{}}],input[{{},{3},{},{}}])) * self.penalty * 0.05
    else
      output[{{},{1},{},{}}] = (1 - torch.cmul(input[{{},{1},{},{}}],input[{{},{2},{},{}}])) * self.penalty 
    end
    
    if self.sizeAverage then
      output = norm * output:sum() 
    else
      output = output:sum() 
    end
    
    return output
end

function OcclusionPriorCriterion:updateGradInput(input, target)
  	assert( input:size(3) == target:size(3) and  input:size(4) == target:size(4),
      "input and target size mismatch")
      
    local norm = input:size(2) / input:nElement();
    
    local gradInput = input:clone()
    
    if input:size(2) == 3 then
      gradInput[{{},{1},{},{}}] = (1 - input[{{},{2},{},{}}]) * self.penalty * 0.05
      gradInput[{{},{2},{},{}}] = -(input[{{},{1},{},{}}] + input[{{},{3},{},{}}]) * self.penalty * 0.05
      gradInput[{{},{3},{},{}}] = (1 - input[{{},{2},{},{}}]) * self.penalty * 0.05
    else
      gradInput[{{},{1},{},{}}] = (1 - input[{{},{2},{},{}}]) * self.penalty
      gradInput[{{},{2},{},{}}] = (1 - input[{{},{1},{},{}}]) * self.penalty
    end
    
    if self.sizeAverage then
      gradInput:mul(norm)
    end

    return gradInput
end