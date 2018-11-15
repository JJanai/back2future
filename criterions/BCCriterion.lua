----------------------------------------------------
---- BRIGHTNESS CONSTANCY CRITERION
-----------------------------------------------------
-- Computes the photometric error using quadratic, L1 or Lorentzian function.
--
-- input -> warped future or past frame (batchSize x ChannelSize x Height x Width)
-- target -> reference frame (batchSize x ChannelSize x Height x Width)
--
-- Copyright 2018 Joel Janai, Fatma Güney, Anurag Ranjan and the Max Planck Gesellschaft.
-- All rights reserved.
-- This software is provided for research purposes only.
-- By using this software you agree to the terms of the license file
-- in the root folder.
---------------------------------------------------------------

require 'criterions.penalty.quadratic_function'

local BCCriterion, parent = torch.class('nn.BCCriterion', 'nn.Criterion')

function BCCriterion:__init()
	parent.__init(self)
  self.sizeAverage = true     -- normalize by number of pixels if true
  self.p = QuadraticPenalty() -- penalty function
end

function BCCriterion:updateOutput(input, target)
	assert( input:nElement() == target:nElement(), "input and target size mismatch")

  local norm = 1.0 / input:nElement()

  -- intensity difference
  local buffer = torch.add(input, -1, target)
  
  -- penalty function
  return norm * self.p:apply(buffer):sum()    
end

function BCCriterion:updateGradInput(input, target)

	assert( input:nElement() == target:nElement(), "input and target size mismatch")
  
  local norm = 1.0 / input:nElement() 

  -- intensity difference
  local buffer = torch.add(input, -1, target)
  
  -- derivative of penalty function
  return self.p:der(self.buffer):mul(norm)
end