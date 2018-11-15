----------------------------------------------------
---- CONSTANT VELOCITY CRITERION
-----------------------------------------------------
-- Enforces a constant velocity on future and past flow field.
-- Computes the average endpoint error between two flow fields.
--
-- input[1] -> batchSize x 2 x Height x Width
-- input[2] -> batchSize x 2 x Height x Width
--
-- Copyright 2018 Joel Janai, Fatma Güney, Anurag Ranjan and the Max Planck Gesellschaft.
-- All rights reserved.
-- This software is provided for research purposes only.
-- By using this software you agree to the terms of the license file
-- in the root folder.
---------------------------------------------------------------

require 'criterions.penalty.quadratic_function'

local ConstVelCriterion, parent = torch.class('nn.ConstVelCriterion', 'nn.Criterion')

function ConstVelCriterion:__init()
  parent.__init(self)
  self.sizeAverage = true
  self.gradCheck = false
end

local  eps = 1e-12

function ConstVelCriterion:updateOutput(input)
  assert( input[1]:nElement() == input[2]:nElement(), "input and target size mismatch")

  local buffer = self.buffer
  local norm = 1.0 / input[1]:nElement();

  -- end point error between flow vectors
  self.output = torch.add(input[1], -1, input[2]):pow(2)
  self.output = torch.sum(self.output,2):sqrt()   -- second channel is flow
  self.output = self.output:sum()

  -- normalization
  if self.sizeAverage then
    self.output = norm * self.output
  end

  return self.output
end

function ConstVelCriterion:updateGradInput(input)

  assert( input[1]:nElement() == input[2]:nElement(),
    "input and target size mismatch")

  local npixels
  local loss

  npixels = input[1]:nElement() / input[1]:size(2)

  -- derivative of end point error
  local buffer = torch.add(input[1], -1, input[2]):pow(2)
  loss = torch.sum(buffer,2):sqrt():add(eps)  -- forms the denominator
  loss = loss:repeatTensor(1,input[1]:size(2),1,1)   -- Repeat tensor to scale the gradients

  local gradInput = {}
  gradInput[1] = torch.add(input[1], -1, input[2]):cdiv(loss)
  gradInput[2] = torch.add(input[2], -1, input[1]):cdiv(loss)

  -- normalization
  if self.sizeAverage then
    gradInput[1] = gradInput[1] / npixels
    gradInput[2] = gradInput[2] / npixels
  end

  return gradInput
end

function ConstVelCriterion:clear()
  self.output = nil
  self.gradInput = nil
end


