----------------------------------------------------
---- KL DIVERGENCE CRITERION
-----------------------------------------------------
-- Computes symmetric KLDivergence between neighboring pixels in occlusion mask.
-- Weights the loss according to image gradients in reference image.
--
-- input -> occlusion masks (batchSize x ChannelSize x Height x Width)
-- target -> reference frame (batchSize x ChannelSize x Height x Width)
--
-- Copyright 2018 Joel Janai, Fatma Güney, Anurag Ranjan and the Max Planck Gesellschaft.
-- All rights reserved.
-- This software is provided for research purposes only.
-- By using this software you agree to the terms of the license file
-- in the root folder.
---------------------------------------------------------------

require 'criterions.penalty.quadratic_function'

local KLDivergenceCriterion, parent = torch.class('nn.KLDivergenceCriterion', 'nn.Criterion')

-- Computes symmetric KLDivergence between neighbors
-- [p * (log(p) - log(q))] + [q * (log(q) - log(p))]

function KLDivergenceCriterion:__init()
  parent.__init(self)
  self.sizeAverage = true
  self.p = QuadraticPenalty()
  self.eps = 5.e-2  -- avoid too large gradients, NANs and INFs
  self.cs = 20
  self.padding = nn.SpatialReplicationPadding(1, 1, 1, 1)
end

function KLDivergenceCriterion:updateOutput(input, target)
  assert( input:size(3) == target:size(3) and input:size(4) == target:size(4),
    "input and target size mismatch")

  local norm = input:size(2) / input:nElement();

  -- avoid nans and infs
  local nonzero = self.padding:forward(input)
  nonzero[torch.le(nonzero,self.eps)] = self.eps

  local log_nonzero = torch.log(nonzero)

  -- symmetric kl divergence [p * (log(p) - log(q))] + [q * (log(q) - log(p))]
  self.gy = (log_nonzero[{{},{},{2,1+input:size(3)},{2,1+input:size(4)}}] - log_nonzero[{{},{},{3,2+input:size(3)},{2,1+input:size(4)}}]):cmul(nonzero[{{},{},{2,1+input:size(3)},{2,1+input:size(4)}}])
  self.gy:add((log_nonzero[{{},{},{3,2+input:size(3)},{2,1+input:size(4)}}] - log_nonzero[{{},{},{2,1+input:size(3)},{2,1+input:size(4)}}]):cmul(nonzero[{{},{},{3,2+input:size(3)},{2,1+input:size(4)}}]))
  self.gx = (log_nonzero[{{},{},{2,1+input:size(3)},{2,1+input:size(4)}}] - log_nonzero[{{},{},{2,1+input:size(3)},{3,2+input:size(4)}}]):cmul(nonzero[{{},{},{2,1+input:size(3)},{2,1+input:size(4)}}])
  self.gx:add((log_nonzero[{{},{},{2,1+input:size(3)},{3,2+input:size(4)}}] - log_nonzero[{{},{},{2,1+input:size(3)},{2,1+input:size(4)}}]):cmul(nonzero[{{},{},{2,1+input:size(3)},{3,2+input:size(4)}}]))

  -- image gradients
  local igy = torch.Tensor(target:size()):zero():cuda()
  local igx = torch.Tensor(target:size()):zero():cuda()
  igy[{{},{},{1,target:size(3)-1},{}}]:add(target[{{},{},{2,target:size(3)},{}}], -1, target[{{},{},{1,target:size(3)-1},{}}])
  igx[{{},{},{},{1,target:size(4)-1}}]:add(target[{{},{},{},{2,target:size(4)}}], -1, target[{{},{},{},{1,target:size(4)-1}}])
  -- contrast sensitive weights
  self.wy = torch.expandAs(torch.exp(-self.cs * torch.mean(torch.abs(igy), 2)), self.gy)
  self.wx = torch.expandAs(torch.exp(-self.cs * torch.mean(torch.abs(igx), 2)), self.gx)

  -- weighted kl divergence between neighbors in x and y direction
  local buffer = torch.add(self.gx:cmul(self.wx), self.gy:cmul(self.wy)):sum()

  -- normalization
  local output
  if self.sizeAverage then
    output = norm * buffer
  else
    output = buffer
  end

  return output
end

function KLDivergenceCriterion:updateGradInput(input, target)
  assert( input:size(3) == target:size(3) and  input:size(4) == target:size(4),
    "input and target size mismatch")

  -- avoid nans and infs
  local nonzero = self.padding:forward(input)
  nonzero[torch.le(nonzero,self.eps)] = self.eps

  local log_nonzero = torch.log(nonzero)

  local gradInput = self.gradInput
  local norm = input:size(2) / input:nElement()

  -- derivatives for symmetric kl divergence
  -- d/dp [p * (log(p) - log(q))] + [q * (log(q) - log(p))] =  log(p) - log(q) + 1 - q/p - p^(-1) / p + log(p) - log(p^(-1)) + 1
  self.gy = (log_nonzero[{{},{},{2,1+input:size(3)},{2,1+input:size(4)}}] - log_nonzero[{{},{},{3,2+input:size(3)},{2,1+input:size(4)}}] + 1
    - torch.cdiv(nonzero[{{},{},{3,2+input:size(3)},{2,1+input:size(4)}}],nonzero[{{},{},{2,1+input:size(3)},{2,1+input:size(4)}}])):cmul(self.wy)
  self.gx = (log_nonzero[{{},{},{2,1+input:size(3)},{2,1+input:size(4)}}] - log_nonzero[{{},{},{2,1+input:size(3)},{3,2+input:size(4)}}] + 1
    - torch.cdiv(nonzero[{{},{},{2,1+input:size(3)},{3,2+input:size(4)}}],nonzero[{{},{},{2,1+input:size(3)},{2,1+input:size(4)}}])):cmul(self.wx)

  local tmp = -torch.cdiv(nonzero[{{},{},{1,input:size(3)},{2,1+input:size(4)}}],nonzero[{{},{},{2,1+input:size(3)},{2,1+input:size(4)}}])
    + log_nonzero[{{},{},{2,1+input:size(3)},{2,1+input:size(4)}}] - log_nonzero[{{},{},{1,input:size(3)},{2,1+input:size(4)}}] + 1
  tmp[{{},{},{2,input:size(3)},{}}]:cmul(self.wy[{{},{},{1,input:size(3)-1},{}}])
  self.gy:add(tmp)

  tmp = -torch.cdiv(nonzero[{{},{},{2,1+input:size(3)},{1,input:size(4)}}],nonzero[{{},{},{2,1+input:size(3)},{2,1+input:size(4)}}])
    + log_nonzero[{{},{},{2,1+input:size(3)},{2,1+input:size(4)}}] - log_nonzero[{{},{},{2,1+input:size(3)},{1,input:size(4)}}] + 1
  tmp[{{},{},{},{2,input:size(4)}}]:cmul(self.wx[{{},{},{},{1,input:size(4)-1}}])
  self.gx:add(tmp)

  -- normalization
  gradInput:resizeAs(input)
  if self.sizeAverage then
    gradInput = norm * ((self.gx) + (self.gy))
  else
    gradInput = ((self.gx) + (self.gy))
  end

  return gradInput
end

function KLDivergenceCriterion:clear()
  self.gy = nil
  self.gx = nil
  self.wy = nil
  self.wx = nil
end

