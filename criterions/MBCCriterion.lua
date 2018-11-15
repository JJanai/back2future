----------------------------------------------------
---- MULTI-FRAME BRIGHTNESS CONSTANCY CRITERION
-----------------------------------------------------
-- Computes the photometric error using quadratic, L1 or Lorentzian function
-- between all warped frames and the reference frame WITHOUT MASKING THE OCCLUSIONS.
-- updateGradInput returns a table of gradients for the input table
--
-- input -> table consisting of the {flow_future, flow_past, occlusion, warped_img_1, ..., warped_img_N}
--  flow_x -> flow to future or past (batchSize x 2 x Height x Width)
--  occlusion -> occlusions (batchSize x 2 x Height x Width)
--  warped_img_i -> warped image i (batchSize x ChannelSize x Height x Width)
-- target -> reference frame (batchSize x ChannelSize x Height x Width)
--
-- Copyright 2018 Joel Janai, Fatma Güney, Anurag Ranjan and the Max Planck Gesellschaft.
-- All rights reserved.
-- This software is provided for research purposes only.
-- By using this software you agree to the terms of the license file
-- in the root folder.
---------------------------------------------------------------

require 'criterions.penalty.quadratic_function'

local MBCCriterion, parent = torch.class('nn.MBCCriterion', 'nn.Criterion')

function MBCCriterion:__init()
  parent.__init(self)
  self.sizeAverage = true     -- normalize by number of pixels if true
  self.p = QuadraticPenalty() -- penalty function
  self.F = 3                  -- number of frames
  self.gradCheck = false      -- check gradients
  self.pwc_flow_scaling = 1   -- flow scaling factor
  self.backward_flow = false  -- true if backward flow is computed
end

function MBCCriterion:updateOutput(input, target)
  assert( #input >= 2, "expecting at least two inputs")

  -- set start of first warped frame
  local warp_start = 3
  if self.F == 2 then
    warp_start = 2
  end
  if self.backward_flow then
    warp_start = warp_start + 1
  end

  local norm = input[warp_start]:size(2) / (input[warp_start]:nElement())

  local ref = 0.5 * (self.F - 1)  -- reference frame
  local b = input[1]:size(1)      -- batches
  local h = input[1]:size(3)      -- height
  local w = input[1]:size(4)      -- width

  -- create image with pixel locations for out of image check
  self.coord = input[1]:clone()
  self.coord[{{},{1},{},{}}] = torch.range(1,w):repeatTensor(b,1,h,1)                 -- x coordinate
  self.coord[{{},{2},{},{}}] = torch.range(1,h):repeatTensor(b,1,w,1):transpose(3,4)  -- y coordinate

  local acc = input[warp_start][{{},{1},{},{}}]:clone():fill(0)
  for f = 1,(self.F - 1) do
    local img = input[warp_start - 1 + f]

    assert( img:nElement() == target:nElement(), "input and target size mismatch")

    local buffer = torch.add(img, -1, target)
    buffer = torch.sum(self.p:apply(buffer), 2)

    if self.gradCheck == false then        
      -- compute target locations
      local tcoord
      if self.F == 2 then
        tcoord = self.coord + input[1] * self.pwc_flow_scaling
      elseif f <= ref then
        if self.backward_flow then
          tcoord = self.coord + (f - ref - 1) * input[2] * self.pwc_flow_scaling
        else
          tcoord = self.coord + (f - ref - 1) * input[1] * self.pwc_flow_scaling
        end
      else
        tcoord = self.coord + (f - ref) * input[1] * self.pwc_flow_scaling
      end
      
      -- mask pixels out of image
      local mask = torch.ge(tcoord[{{},{1},{},{}}], 1) -- left
      mask:cmul(torch.ge(tcoord[{{},{2},{},{}}], 1))  -- top
      mask:cmul(torch.le(tcoord[{{},{1},{},{}}],w)) -- right
      mask:cmul(torch.le(tcoord[{{},{2},{},{}}],h)) -- bottom
      mask = mask:cuda()
      buffer:cmul(mask)
    end

    acc:add(buffer)
  end

  -- normalization
  local output = acc:sum() / (input[warp_start]:size(2) * (self.F - 1))
  if self.sizeAverage then
    output = norm * output
  end

  return output
end

function MBCCriterion:updateGradInput(input, target)
  assert( #input >= 2, "expecting at least two inputs")

  local ref = 0.5 * (self.F - 1)  -- reference frame
  local b = input[1]:size(1)      -- batches
  local h = input[1]:size(3)      -- height
  local w = input[1]:size(4)      -- width

  -- set start of first warped frame
  local warp_start = 3
  local gradSize = self.F
  if self.F == 2 then
    warp_start = 2
    gradSize = gradSize - 1
  end
  
  -- set gradient start (only warped images)
  local w_g_start = warp_start - 2
  
  -- backward flow
  if self.backward_flow then
    warp_start = warp_start + 1
  end

  local norm = input[warp_start]:size(2) / (input[warp_start]:nElement())

  local gradInput = {}
  for f = 1,gradSize do
    table.insert(gradInput, input[warp_start-2+f].new())
  end

  if self.F > 2 then
    gradInput[1]:resizeAs(input[warp_start - 1]):fill(0) -- occlusion
  end
  
  -- iterate over all warped images
  for f = 1,(self.F - 1) do
    local img = input[warp_start - 1 + f]

    assert( img:nElement() == target:nElement(), "input and target size mismatch")

    -- compute photometric error
    local buffer = torch.add(img, -1, target)

    gradInput[w_g_start+f]:resizeAs(buffer)
    gradInput[w_g_start+f]:copy(self.p:der(buffer))

    if self.gradCheck == false then
      -- compute target pixel location
      local tcoord
      if self.F == 2 then
        tcoord = self.coord + input[1] * self.pwc_flow_scaling
      elseif f <= ref then
        if self.backward_flow then
          tcoord = self.coord + (f - ref - 1) * input[2] * self.pwc_flow_scaling
        else
          tcoord = self.coord + (f - ref - 1) * input[1] * self.pwc_flow_scaling
        end
      else
        tcoord = self.coord + (f - ref) * input[1] * self.pwc_flow_scaling
      end

      -- mask for out of image pixels
      local mask = torch.ge(tcoord[{{},{1},{},{}}], 1) -- left
      mask:cmul(torch.ge(tcoord[{{},{2},{},{}}], 1))  -- top
      mask:cmul(torch.le(tcoord[{{},{1},{},{}}],w)) -- right
      mask:cmul(torch.le(tcoord[{{},{2},{},{}}],h)) -- bottom
      mask = mask:cuda()

      -- mask pixels for image gradients
      mask = torch.repeatTensor(mask,1,input[warp_start]:size(2),1,1)
      gradInput[w_g_start+f]:cmul(mask)
    end

    -- normalization
    gradInput[w_g_start + f]:mul(1/(input[warp_start]:size(2) * (self.F - 1)))
    if self.sizeAverage then
      gradInput[w_g_start+f]:mul(norm)
    end
  end

  return gradInput
end

function MBCCriterion:clear()
  self.buffer = nil
  self.mask = nil
  self.coord = nil
end
