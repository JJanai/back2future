----------------------------------------------------
---- MULTI-FRAME STRUCTURE SIMILARITY MEASURE AND PHOTOMETRIC CRITERION
-----------------------------------------------------
-- Computes the structure similarity measure and photometric L1 loss
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

require 'criterions.penalty.L1_function'

local MSSIML1Criterion, parent = torch.class('nn.MSSIML1Criterion', 'nn.Criterion')

function MSSIML1Criterion:__init()
  parent.__init(self)
  self.sizeAverage = true     -- normalize by number of pixels if true
  self.p = L1Penalty()        -- penalty function
  self.F = 3                  -- number of frames
  self.L = 1                  -- dynamic range of pixel values
  self.alpha = 0.85           -- weight of SSIM
  self.gradCheck = false      -- check gradients
  self.pwc_flow_scaling = 1   -- flow scaling factor
  self.past_flow = false  -- true if past flow is computed
  -- use gaussian to compute expected value
  self.kernel = image.gaussian{size = 3, normalize = true}
  self.conv = nn.Sequential():add(nn.SpatialReplicationPadding(1, 1, 1, 1)):add(nn.SpatialConvolution(3, 3, 3, 3, 1, 1, 0, 0)) -- add padding
  self.conv:get(2).weight:zero()
  self.conv:get(2).bias:zero()
  self.conv:get(2).weight[{{1},{1},{},{}}]:copy(self.kernel)
  self.conv:get(2).weight[{{2},{2},{},{}}]:copy(self.kernel)
  self.conv:get(2).weight[{{3},{3},{},{}}]:copy(self.kernel)
end

function MSSIML1Criterion:updateOutput(input, y)
  assert( #input >= 2, "expecting at least two inputs")

  local ref = 0.5 * (self.F - 1)  -- reference frame
  local b = input[1]:size(1)      -- batches
  local h = input[1]:size(3)      -- height
  local w = input[1]:size(4)      -- width

  -- set start of first warped frame
  local warp_start = 3
  if self.F == 2 then
    warp_start = 2
  end
  if self.past_flow then
    warp_start = warp_start + 1
  end
  
  -- get max and minimum intensity for normalization
  self.mx = torch.max(y)
  self.mn = torch.min(y)
  for i = 2,#input do
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

  -- create image with pixel locations for out of image check
  self.coord = input[1]:clone()
  self.coord[{{},{1},{},{}}] = torch.range(1,w):repeatTensor(b,1,h,1)                 -- x coordinate
  self.coord[{{},{2},{},{}}] = torch.range(1,h):repeatTensor(b,1,w,1):transpose(3,4)  -- y coordinate

  -- stabilization of the division with weak denominator
  local C1 = torch.pow(0.01*self.L, 2)
  local C2 = torch.pow(0.03*self.L, 2)

  -- compute mean and variance in target
  self.mu_y = self.conv:forward(self.target):clone()
  self.sigma_y = self.conv:forward(torch.pow(self.target, 2)):clone() - torch.pow(self.mu_y, 2)

  -- accumulate errors
  local acc = torch.Tensor(b,1,h,w):zero()
  if self.gradCheck == false then
    acc = acc:cuda()
  end
  
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

    -- compute target pixel location
    local tcoord
    if self.F == 2 then
      tcoord = self.coord + input[1] * self.pwc_flow_scaling
    elseif f <= ref then
      if self.past_flow then
        tcoord = self.coord + (f - ref - 1) * input[2] * self.pwc_flow_scaling
      else
        tcoord = self.coord + (f - ref - 1) * input[1] * self.pwc_flow_scaling
      end
    else
      tcoord = self.coord + (f - ref) * input[1] * self.pwc_flow_scaling
    end
    
    -- mask pixels out of image    
    if self.gradCheck == false then
      local mask = torch.ge(tcoord[{{},{1},{},{}}], 1) -- left
      mask:cmul(torch.ge(tcoord[{{},{2},{},{}}], 1))  -- top
      mask:cmul(torch.le(tcoord[{{},{1},{},{}}],w)) -- right
      mask:cmul(torch.le(tcoord[{{},{2},{},{}}],h)) -- bottom
      mask = mask:cuda()
      tmp:cmul(mask)
    end

    acc:add(tmp)
  end

  -- normalization
  self.output = acc:sum() / (input[warp_start]:size(2) * (self.F - 1))
  if self.sizeAverage then
    self.output = norm * self.output
  end

  return self.output
end

function MSSIML1Criterion:updateGradInput(input, y)
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
  
  -- past flow
  if self.past_flow then
    warp_start = warp_start + 1
  end
  
  local ch = input[warp_start]:size(2)

  local norm = ch / (input[warp_start]:nElement())

  local gradInput = {}
  for f = 1,gradSize do
    table.insert(gradInput, input[warp_start-2+f].new())
  end

  -- stabilization of the division with weak denominator
  local C1 = torch.pow(0.01*self.L, 2)
  local C2 = torch.pow(0.03*self.L, 2)

  -- no occlusions for 2 frames
  if self.F > 2 then
    gradInput[1]:resizeAs(input[warp_start - 1]):fill(0) -- occlusion
  end
  
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
    gradInput[w_g_start + f]:resizeAs(buffer)
    gradInput[w_g_start + f]:copy(-self.alpha * (torch.cmul(d_SSIM_l, SSIM_cs) + torch.cmul(SSIM_l, d_SSIM_cs))  + (1 - self.alpha) * self.p:der(buffer))

    -- compute target pixel location
    local tcoord
    if self.F == 2 then
      tcoord = self.coord + input[1] * self.pwc_flow_scaling
    elseif f <= ref then
      if self.past_flow then
        tcoord = self.coord + (f - ref - 1) * input[2] * self.pwc_flow_scaling
      else
        tcoord = self.coord + (f - ref - 1) * input[1] * self.pwc_flow_scaling
      end
    else
      tcoord = self.coord + (f - ref) * input[1] * self.pwc_flow_scaling
    end
    
    -- mask for out of image pixels
    if self.gradCheck == false then
      local mask = torch.ge(tcoord[{{},{1},{},{}}], 1) -- left
      mask:cmul(torch.ge(tcoord[{{},{2},{},{}}], 1))  -- top
      mask:cmul(torch.le(tcoord[{{},{1},{},{}}],w)) -- right
      mask:cmul(torch.le(tcoord[{{},{2},{},{}}],h)) -- bottom
      mask = mask:cuda()

      -- mask pixels for image gradients
      mask = torch.repeatTensor(mask,1,ch,1,1)
      gradInput[w_g_start + f]:cmul(mask)
    end

    -- normalization
    gradInput[w_g_start + f]:mul(1/(input[warp_start]:size(2) * (self.F - 1)))
    if self.sizeAverage then
      gradInput[w_g_start + f]:mul(norm)
    end
  end

  return gradInput
end

function MSSIML1Criterion:clear()
  self.coord = nil
end

