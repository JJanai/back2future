----------------------------------------------------
---- OCCLUSION AWARE MULTI-FRAME BRIGHTNESS CONSTANCY CRITERION
-----------------------------------------------------
-- Computes the photometric error using quadratic, L1 or Lorentzian function
-- between all warped frames and the reference frame WHILE MASKING THE OCCLUSIONS.
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

local OBCCriterion, parent = torch.class('nn.OBCCriterion', 'nn.Criterion')

function OBCCriterion:__init()
  parent.__init(self)
  self.sizeAverage = true     -- normalize by number of pixels if true
  self.gradCheck = false      -- check gradients
  self.p = QuadraticPenalty() -- penalty function
  self.penalty_out = 1.0      -- penalty for out of image pixels
  self.F = 3                  -- number of frames
  self.pwc_flow_scaling = 1   -- flow scaling factor
  self.past_flow = false  -- true if past flow is computed
end

function OBCCriterion:updateOutput(input, target)
  assert( #input >= 4, "expecting at least four inputs")

  -- set start of first warped frame
  local warp_start = 3
  if self.past_flow then
    warp_start = 4
  end

  local tmp
  local norm = input[warp_start]:size(2) / (input[warp_start]:nElement())

  local ref = 0.5 * (self.F - 1)  -- reference frame
  local b = input[1]:size(1)      -- batches
  local h = input[1]:size(3)      -- height
  local w = input[1]:size(4)      -- width

  -- create image with pixel locations for out of image check
  self.coord = input[1]:clone()
  self.coord[{{},{1},{},{}}] = torch.range(1,w):repeatTensor(b,1,h,1)                 -- x coordinate
  self.coord[{{},{2},{},{}}] = torch.range(1,h):repeatTensor(b,1,w,1):transpose(3,4)  -- y coordinate

  -- accumulate errors
  local acc = torch.Tensor(b,1,h,w):zero()
  if self.gradCheck == false then
    acc = acc:cuda()
  end

  -- get occlusions
  local occ = input[warp_start-1]
  
  -- iterate over all warped images
  for f = 1,(self.F - 1) do
    local img = input[warp_start - 1 + f]

    assert( img:nElement() == target:nElement(), "input and target size mismatch")

    -- compute photometric error
    local buffer = torch.add(img, -1, target)
    tmp = torch.sum(self.p:apply(buffer), 2)

    -- compute target locations and mask occlusions
    local tcoord
    if f <= ref then
      if self.past_flow then
        tcoord = self.coord + (f - ref - 1) * input[2] * self.pwc_flow_scaling
      else
        tcoord = self.coord + (f - ref - 1) * input[1] * self.pwc_flow_scaling
      end

      local tocc = (occ[{{},{2},{},{}}]) -- Visible or future occluded
      tmp:cmul(tocc)
    else
      tcoord = self.coord + (f - ref) * input[1] * self.pwc_flow_scaling

      local tocc = (occ[{{},{1},{},{}}]) -- Visible or past occluded
      tmp:cmul(tocc)
    end

    -- mask pixels out of image
    if self.gradCheck == false then
      local mask = torch.ge(tcoord[{{},{1},{},{}}], 1) -- left
      mask:cmul(torch.ge(tcoord[{{},{2},{},{}}], 1))  -- top
      mask:cmul(torch.le(tcoord[{{},{1},{},{}}],w)) -- right
      mask:cmul(torch.le(tcoord[{{},{2},{},{}}],h)) -- bottom
      mask = mask:cuda()
      tmp:cmul(mask)

      -- add out of image penalty
      local pen = (1 - mask) * self.penalty_out
      tmp:add(pen)
    end

    acc:add(tmp)
  end

  -- normalize
  self.output = acc:sum() / (input[warp_start]:size(2) * (self.F - 1))
  if self.sizeAverage then
    self.output = norm * self.output
  end

  return self.output
end

function OBCCriterion:updateGradInput(input, target)
  assert( #input >= 4, "expecting at least four inputs")

  -- set start of first warped frame
  local warp_start = 3
  if self.past_flow then
    warp_start = 4
  end

  local norm = input[warp_start]:size(2) / (input[warp_start]:nElement());

  local gradInput = {}
  for f = 1,self.F do
    table.insert(gradInput, input[warp_start-2+f].new())
  end

  local ref = 0.5 * (self.F - 1)  -- reference frame
  local b = input[1]:size(1)      -- batches
  local h = input[1]:size(3)      -- height
  local w = input[1]:size(4)      -- width

  local occ = input[warp_start-1]

  gradInput[1]:resizeAs(input[warp_start-1]):fill(0)

  -- iterate over all warped images
  for f = 1,(self.F - 1) do
    local img = input[warp_start - 1 + f]

    assert( img:nElement() == target:nElement(), "input and target size mismatch")

    -- photometric error
    local buffer = torch.add(img, -1, target)

    -- compute derivative for frame f
    gradInput[1 + f]:resizeAs(buffer)
    gradInput[1 + f]:copy(self.p:der(buffer))

    -- mask occlusions and compute derivative for occlusions
    buffer = torch.sum(self.p:apply(buffer), 2)
    if f <= ref then
      -- for past frames
      if self.gradCheck == false then
        -- compute target pixel location
        local tcoord
        if self.past_flow then
          tcoord = self.coord + (f - ref - 1) * input[2] * self.pwc_flow_scaling
        else
          tcoord = self.coord + (f - ref - 1) * input[1] * self.pwc_flow_scaling
        end

        -- mask for out of image pixels
        local mask = torch.ge(tcoord[{{},{1},{},{}}], 1) -- left
        mask:cmul(torch.ge(tcoord[{{},{2},{},{}}], 1))  -- top
        mask:cmul(torch.le(tcoord[{{},{1},{},{}}],w)) -- right
        mask:cmul(torch.le(tcoord[{{},{2},{},{}}],h)) -- bottom
        mask = mask:cuda()

        -- mask future occlusions gradients and add occlusion penalty
        buffer:cmul(mask)
        local pen = (1 - mask) * self.penalty_out
        buffer:add(pen)

        -- mask pixels for image gradients
        mask = torch.repeatTensor(mask,1,input[warp_start]:size(2),1,1)
        gradInput[1 + f]:cmul(mask)
      end

      -- future occlusions gradients
      gradInput[1][{{},{2},{},{}}]:add(buffer)

      -- mask occlusions
      local tocc = (occ[{{},{2},{},{}}]):repeatTensor(1,input[warp_start]:size(2),1,1) -- Visible or future occluded
      gradInput[1 + f]:cmul(tocc)
    else
      -- for future frames
      if self.gradCheck == false then
        -- compute target pixel location
        local tcoord = self.coord + (f - ref) * input[1] * self.pwc_flow_scaling

        -- mask for out of image pixels
        local mask = torch.ge(tcoord[{{},{1},{},{}}], 1) -- left
        mask:cmul(torch.ge(tcoord[{{},{2},{},{}}], 1))  -- top
        mask:cmul(torch.le(tcoord[{{},{1},{},{}}],w)) -- right
        mask:cmul(torch.le(tcoord[{{},{2},{},{}}],h)) -- bottom
        mask = mask:cuda()

        -- mask past occlusions gradients and add occlusion penalty
        buffer:cmul(mask)
        local pen = (1 - mask) * self.penalty_out
        buffer:add(pen)

        -- mask pixels for image gradients
        mask = torch.repeatTensor(mask,1,input[warp_start]:size(2),1,1)
        gradInput[1 + f]:cmul(mask)
      end

      -- past occlusions gradients
      gradInput[1][{{},{1},{},{}}]:add(buffer)

      -- mask occlusions
      local tocc = (occ[{{},{1},{},{}}]):repeatTensor(1,input[warp_start]:size(2),1,1)-- Visible or past occluded
      gradInput[1 + f]:cmul(tocc)
    end

    -- normalization of image gradients
    gradInput[1 + f]:mul(1/(input[warp_start]:size(2) * (self.F - 1)))
    if self.sizeAverage then
      gradInput[1 + f]:mul(norm)
    end
  end

  -- normalization of occlusions gradients
  gradInput[1]:mul(1/(input[warp_start]:size(2) * (self.F - 1)))
  if self.sizeAverage then
    gradInput[1]:mul(norm)
  end

  return gradInput
end

function OBCCriterion:clear()
  self.coord = nil
end
