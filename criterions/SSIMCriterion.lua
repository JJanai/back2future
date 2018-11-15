----------------------------------------------------
---- STRUCTURE SIMILARITY MEASURE CRITERION
-----------------------------------------------------
-- Computes the structure similarity measure between the warped
-- image and the reference image.
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

require 'image'
require 'nn'
require 'cunn'
require 'cudnn'

local SSIMCriterion, parent = torch.class('nn.SSIMCriterion', 'nn.Criterion')

function SSIMCriterion:__init()
	parent.__init(self)
	self.sizeAverage = true   -- normalize by number of pixels if true
  self.L = 1                -- dynamic range of pixel values
  self.gradCheck = false    -- check gradients
  -- use gaussian to compute expected value
  self.kernel = image.gaussian{size = 3, normalize = true}
  self.conv = nn.Sequential():add(nn.SpatialReplicationPadding(1, 1, 1, 1)):add(nn.SpatialConvolution(3, 3, 3, 3, 1, 1, 0, 0)) -- add padding
  self.conv:get(2).weight:zero()
  self.conv:get(2).bias:zero()
  self.conv:get(2).weight[{{1},{1},{},{}}]:copy(self.kernel)
  self.conv:get(2).weight[{{2},{2},{},{}}]:copy(self.kernel)
  self.conv:get(2).weight[{{3},{3},{},{}}]:copy(self.kernel)
end

function SSIMCriterion:updateOutput(input, target)
  assert( input:nElement() == target:nElement(), "input and target size mismatch")
  
  -- get max and minimum intensity for normalization
	local x = input:clone()
	local y = target:clone()
  local mx = torch.max(torch.cat(x, y, 2))
  local mn = torch.min(torch.cat(x, y, 2))
  
  if self.gradCheck then
    mx = 1
    mn = 0
  end
  
  x = (x - mn) / (mx - mn)
  y = (y - mn) / (mx - mn)
  
  local b = input:size(1)

  -- stabilization of the division with weak denominator
  local C1 = torch.pow(0.01*self.L, 2)
  local C2 = torch.pow(0.03*self.L, 2)
  local norm = (self.sizeAverage and 1.0 / x:nElement() or 1.);
  
  -- compute mean
  local mu_x = self.conv:forward(x):clone()
  local mu_y = self.conv:forward(y):clone()
  
  -- compute variance and covariance
  local sigma_x = self.conv:forward(torch.pow(x,2)):clone() - torch.pow(mu_x, 2)
  local sigma_y = self.conv:forward(torch.pow(y, 2)):clone() - torch.pow(mu_y, 2)
  local sigma_xy = self.conv:forward(torch.cmul(x, y)):clone() - torch.cmul(mu_x,mu_y)

  -- compute luminance, contrast and strucuture
  local SSIM_l = torch.cdiv(2 * torch.cmul(mu_x,mu_y) + C1, torch.pow(mu_x, 2) + torch.pow(mu_y, 2) + C1)
  local SSIM_cs = torch.cdiv(2 * sigma_xy + C2, sigma_x + sigma_y + C2)
 
  -- normalization
  return norm * (0.5*(1 - torch.cmul(SSIM_l,SSIM_cs))):sum()
end

function SSIMCriterion:updateGradInput(input, target)
  assert( input:nElement() == target:nElement(), "input and target size mismatch")
  
  local x = input:clone()
  local y = target:clone()
  local mx = torch.max(torch.cat(x, y, 2))
  local mn = torch.min(torch.cat(x, y, 2))
  
  if self.gradCheck then
    mx = 1
    mn = 0
  end
  
  x = (x - mn) / (mx - mn)
  y = (y - mn) / (mx - mn)

  -- stabilization of the division with weak denominator
  local C1 = torch.pow(0.01*self.L, 2)
  local C2 = torch.pow(0.03*self.L, 2)
  local norm = (self.sizeAverage and 1.0 / x:nElement() or 1.);
  
  local n = 0.5 * (self.kernel:size(1) - 1)
  local b = input:size(1)
  local width = input:size(3)
  local h = input:size(4)
  
  -- compute mean
  local mu_x = self.conv:forward(x):clone()
  local mu_y = self.conv:forward(y):clone()
  
  -- compute variance and covariance
  local sigma_x = self.conv:forward(torch.pow(x,2)):clone() - torch.pow(mu_x, 2)
  local sigma_y = self.conv:forward(torch.pow(y, 2)):clone() - torch.pow(mu_y, 2)
  local sigma_xy = self.conv:forward(torch.cmul(x, y)):clone() - torch.cmul(mu_x,mu_y)

  -- compute luminance, contrast and strucuture
  local SSIM_l = torch.cdiv(2 * torch.cmul(mu_x,mu_y) + C1, torch.pow(mu_x, 2) + torch.pow(mu_y, 2) + C1)
  local SSIM_cs = torch.cdiv(2 * sigma_xy + C2, sigma_x + sigma_y + C2)
  
  -- compute derivatives
  local gw = self.kernel[{{n + 1},{n + 1}}]:squeeze()
  local d_SSIM_l = 2 * gw * torch.cdiv(mu_y - torch.cmul(mu_x, SSIM_l), torch.pow(mu_x, 2) + torch.pow(mu_y, 2) + C1)
  local d_SSIM_cs = 2 * gw * torch.cdiv((y - mu_y) - torch.cmul(SSIM_cs, x - mu_x), sigma_x + sigma_y + C2)
  
  -- mask out border pixels
  local coord = torch.Tensor(b,2,width,h)
  coord[{{},{1},{},{}}] = torch.range(1,width):repeatTensor(b,1,h,1):transpose(3,4)
  coord[{{},{2},{},{}}] = torch.range(1,h):repeatTensor(b,1,width,1)
  local b_mask = torch.gt(coord[{{},{1},{},{}}], 1):cmul(torch.gt(coord[{{},{2},{},{}}], 1)):cmul(torch.lt(coord[{{},{1},{},{}}],width)):cmul(torch.lt(coord[{{},{2},{},{}}],h))
  if not self.gradCheck then
    b_mask:cuda()
  end
  b_mask = torch.repeatTensor(b_mask,1,input:size(2),1,1)
  
  -- normalization
  self.gradInput = -norm * (torch.cmul(d_SSIM_l, SSIM_cs) + torch.cmul(SSIM_l, d_SSIM_cs))
  if not self.gradCheck then
    self.gradInput:cmul(b_mask)
  end
  
  return self.gradInput
end