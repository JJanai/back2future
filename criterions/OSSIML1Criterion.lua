----------------------------------------------------
---- OCCLUSION AWARE MULTI-FRAME STRUCTURE SIMILARITY MEASURE L1 CRITERION
-----------------------------------------------------
-- Computes the structure similarity measure and photometric L1 loss
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

require 'criterions.penalty.L1_function'
require 'image'

local OSSIML1Criterion, parent = torch.class('nn.OSSIML1Criterion', 'nn.Criterion')

function OSSIML1Criterion:__init()
  parent.__init(self)
  self.sizeAverage = true     -- normalize by number of pixels if true
  self.p = L1Penalty()        -- penalty function
  self.penalty_out = 1.0      -- penalty for out of image pixels
  self.F = 3                  -- number of frames
  self.L = 1                  -- dynamic range of pixel values
  self.alpha = 0.85           -- weight of SSIM
  self.gradCheck = false      -- check gradients
  self.pwc_flow_scaling = 1   -- flow scaling factor
  self.backward_flow = false  -- true if backward flow is computed
  -- use gaussian to compute expected value
  self.kernel = image.gaussian{size = 3, normalize = true}
  self.conv = nn.Sequential():add(nn.SpatialReplicationPadding(1, 1, 1, 1)):add(nn.SpatialConvolution(3, 3, 3, 3, 1, 1, 0, 0)) -- add padding
  self.conv:get(2).weight:zero()
  self.conv:get(2).bias:zero()
  self.conv:get(2).weight[{{1},{1},{},{}}]:copy(self.kernel)
  self.conv:get(2).weight[{{2},{2},{},{}}]:copy(self.kernel)
  self.conv:get(2).weight[{{3},{3},{},{}}]:copy(self.kernel)
end

function OSSIML1Criterion:updateOutput(input, y)
  assert( #input >= 4, "expecting at least four inputs")

  local ref = 0.5 * (self.F - 1)  -- reference frame
  local b = input[1]:size(1)      -- batches
  local h = input[1]:size(3)      -- height
  local w = input[1]:size(4)      -- width

  -- set start of first warped frame
  local warp_start = 3
  if self.backward_flow then
    warp_start = 4
  end

  -- get max and minimum intensity for normalization
  self.mx = torch.max(y)
  self.mn = torch.min(y)
  for i = warp_start,#input do
    self.mx = math.max(self.mx, torch.max(input[i]))
    self.mn = math.min(self.mn, torch.min(input[i]))
  end
  if self.gradCheck == true then
    self.mx = 1
    self.mn = 0
  end
  self.target = (y - self.mn) / (self.mx - self.mn)

  local tmp
  local norm = input[warp_start]:size(2) / (input[warp_start]:nElement())

  -- stabilization of the division with weak denominator
  local C1 = torch.pow(0.01*self.L, 2)
  local C2 = torch.pow(0.03*self.L, 2)

  -- occluded, not occluded
  local occ = input[warp_start-1]
  local iocc = 1 - occ

  -- create image with pixel locations for out of image check
  self.coord = input[1]:clone()
  self.coord[{{},{1},{},{}}] = torch.range(1,w):repeatTensor(b,1,h,1)                 -- x coordinate
  self.coord[{{},{2},{},{}}] = torch.range(1,h):repeatTensor(b,1,w,1):transpose(3,4)  -- y coordinate

  -- accumulate error
  local acc = torch.Tensor(b,1,h,w):zero()
  if self.gradCheck == false then
    acc = acc:cuda()
  end

  -- compute mean and variance in target
  self.mu_y = self.conv:forward(self.target):clone()
  self.sigma_y = self.conv:forward(torch.pow(self.target, 2)):clone() - torch.pow(self.mu_y, 2)

  -- iterate over all warped images
  for f = 1,(self.F - 1) do
    local img = (input[warp_start - 1 + f] - self.mn) / (self.mx - self.mn)

    assert( img:nElement() == self.target:nElement(), "input and target size mismatch")

    -- compute photometric error
    local buffer = torch.add(img, -1, self.target)

    -- compute mean, variance and covariance
    local mu_x = self.conv:forward(img):clone()
    local sigma_x = self.conv:forward(torch.pow(img,2)):clone() - torch.pow(mu_x, 2)
    local sigma_xy = self.conv:forward(torch.cmul(img, self.target)):clone() - torch.cmul(mu_x,self.mu_y)

    -- compute luminance, contrast and strucuture
    local SSIM_l = torch.cdiv(2 * torch.cmul(mu_x,self.mu_y) + C1, torch.pow(mu_x, 2) + torch.pow(self.mu_y, 2) + C1)
    local SSIM_cs = torch.cdiv(2 * sigma_xy + C2, sigma_x + self.sigma_y + C2)

    -- weighted sum of (1-SSIM) and brightness constancy
    tmp = self.alpha * (1 - torch.cmul(SSIM_l,SSIM_cs)):sum(2) + (1-self.alpha) * torch.sum(self.p:apply(buffer), 2)

    -- compute target locations and mask occlusions
    local tcoord
    if f <= ref then
      if self.backward_flow then
        tcoord = self.coord + (f - ref - 1) * input[2] * self.pwc_flow_scaling
      else
        tcoord = self.coord + (f - ref - 1) * input[1] * self.pwc_flow_scaling
      end

      local tocc = (occ[{{},{2},{},{}}]) -- Visible or forward occluded
      tmp:cmul(tocc)
    else
      tcoord = self.coord + (f - ref) * input[1] * self.pwc_flow_scaling

      local tocc = (occ[{{},{1},{},{}}]) -- Visible or backward occluded
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

  -- normalization
  local output = acc:sum() / (input[warp_start]:size(2) * (self.F - 1))
  if self.sizeAverage then
    output = norm * output
  end

  return output
end

function OSSIML1Criterion:updateGradInput(input, y)
  assert( #input >= 4, "expecting at least four inputs")

  local ref = 0.5 * (self.F - 1)  -- reference frame
  local b = input[1]:size(1)      -- batches
  local h = input[1]:size(3)      -- height
  local w = input[1]:size(4)      -- width

  -- set start of first warped frame
  local warp_start = 3
  if self.backward_flow then
    warp_start = 4
  end

  local norm = input[warp_start]:size(2) / (input[warp_start]:nElement())

  local gradInput = {}
  for f = 1,self.F do
    table.insert(gradInput, input[warp_start-2+f].new())
  end

  -- stabilization of the division with weak denominator
  local C1 = torch.pow(0.01*self.L, 2)
  local C2 = torch.pow(0.03*self.L, 2)

  local occ = input[warp_start-1]
  local iocc = 1 - occ

  gradInput[1]:resizeAs(input[warp_start-1]):fill(0)

  -- iterate over all warped images
  for f = 1,(self.F - 1) do
    local img = (input[warp_start - 1 + f] - self.mn) / (self.mx - self.mn)

    assert( img:nElement() == self.target:nElement(), "input and target size mismatch")

    -- compute photometric error
    local buffer = torch.add(img, -1, self.target)

    local n = 0.5 * (self.kernel:size(1) - 1)

    -- compute mean, variance and covariance
    local mu_x = self.conv:forward(img):clone()
    local sigma_x = self.conv:forward(torch.pow(img,2)):clone() - torch.pow(mu_x, 2)
    local sigma_xy = self.conv:forward(torch.cmul(img, self.target)):clone() - torch.cmul(mu_x,self.mu_y)

    -- compute luminance, contrast and strucuture
    local SSIM_l = torch.cdiv(2 * torch.cmul(mu_x,self.mu_y) + C1, torch.pow(mu_x, 2) + torch.pow(self.mu_y, 2) + C1)
    local SSIM_cs = torch.cdiv(2 * sigma_xy + C2, sigma_x + self.sigma_y + C2)

    -- compute derivatives
    local gw = self.kernel[{{n + 1},{n + 1}}]:squeeze()
    local d_SSIM_l = 2 * gw * torch.cdiv(self.mu_y - torch.cmul(mu_x, SSIM_l), torch.pow(mu_x, 2) + torch.pow(self.mu_y, 2) + C1)
    local d_SSIM_cs = 2 * gw * torch.cdiv((self.target - self.mu_y) - torch.cmul(SSIM_cs, img - mu_x), sigma_x + self.sigma_y + C2)

    -- weighted sum of derivatives
    gradInput[1 + f]:resizeAs(buffer)
    gradInput[1 + f]:copy(-self.alpha * (torch.cmul(d_SSIM_l, SSIM_cs) + torch.cmul(SSIM_l, d_SSIM_cs))  + (1 - self.alpha) * self.p:der(buffer))

    buffer = self.alpha * (1 - torch.cmul(SSIM_l,SSIM_cs)):sum(2) + (1-self.alpha) * torch.sum(self.p:apply(buffer), 2)

    if f <= ref then
      if self.gradCheck == false then -- compute target pixel location
        local tcoord
        if self.backward_flow then
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

        -- mask forward occlusions gradients and add occlusion penalty
        buffer:cmul(mask)
        local pen = (1 - mask) * self.penalty_out
        buffer:add(pen)

        -- mask pixels for image gradients
        mask = torch.repeatTensor(mask,1,input[warp_start]:size(2),1,1)
        gradInput[1 + f]:cmul(mask)
      end

      -- forward occlusions gradients
      gradInput[1][{{},{2},{},{}}]:add(buffer)

      -- mask occlusions
      local tocc = (occ[{{},{2},{},{}}]):repeatTensor(1,input[warp_start]:size(2),1,1) -- Visible or forward occluded
      gradInput[1 + f]:cmul(tocc)
    else
      if self.gradCheck == false then
        -- compute target pixel location
        local tcoord = self.coord + (f - ref) * input[1] * self.pwc_flow_scaling

        -- mask for out of image pixels
        local mask = torch.ge(tcoord[{{},{1},{},{}}], 1) -- left
        mask:cmul(torch.ge(tcoord[{{},{2},{},{}}], 1))  -- top
        mask:cmul(torch.le(tcoord[{{},{1},{},{}}],w)) -- right
        mask:cmul(torch.le(tcoord[{{},{2},{},{}}],h)) -- bottom
        mask = mask:cuda()

        -- mask forward occlusions gradients and add occlusion penalty
        buffer:cmul(mask)
        local pen = (1 - mask) * self.penalty_out
        buffer:add(pen)

        -- mask pixels for image gradients
        mask = torch.repeatTensor(mask,1,input[warp_start]:size(2),1,1)
        gradInput[1 + f]:cmul(mask)
      end

      -- backward occlusions gradients
      gradInput[1][{{},{1},{},{}}]:add(buffer)

      -- mask occlusions
      local tocc = (occ[{{},{1},{},{}}]):repeatTensor(1,input[warp_start]:size(2),1,1)-- Visible or forward occluded
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

function OSSIML1Criterion:clear()
  self.coord = nil
end
