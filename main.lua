-- Copyright 2018 Joel Janai, Fatma Güney, Anurag Ranjan and the Max Planck Gesellschaft.
-- All rights reserved.
-- This software is provided for research purposes only.
-- By using this software you agree to the terms of the license file
-- in the root folder.
-- For commercial use, please contact ps-license@tue.mpg.de.

require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'

torch.setdefaulttensortype('torch.FloatTensor')

local opts = paths.dofile('opts.lua')
opt = opts.parse(arg)

cutorch.setDevice(opt.GPU) -- by default, use GPU 1
torch.manualSeed(opt.manualSeed)

paths.dofile('util.lua')
paths.dofile('model.lua')

opt.imageSize = model.imageSize or opt.imageSize
opt.outputSize = model.outputSize or opt.outputSize

paths.dofile('data.lua')
paths.dofile('train.lua')
paths.dofile('test.lua')

epoch = opt.epochNumber

if opt.resubmit > 0 then
  opt.nEpochs = opt.resubmit * opt.epochStore
end

for i=1,opt.nEpochs do
  train()
  test()
  epoch = epoch + 1
end
