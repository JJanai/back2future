-- Copyright 2018 Joel Janai, Fatma Güney, Anurag Ranjan and the Max Planck Gesellschaft.
-- All rights reserved.
-- This software is provided for research purposes only.
-- By using this software you agree to the terms of the license file
-- in the root folder.
-- For commercial use, please contact ps-license@tue.mpg.de.

require 'image'
require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'stn'
require 'spy'

local flowX = require 'flowExtensions'
local TF = require 'transforms'
local stringx = require('pl.stringx')

paths.dofile('dataset.lua')
paths.dofile('util.lua')

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------
local eps = 1e-6
-- a cache file of the training metadata (if doesnt exist, will be created)
local trainCache = paths.concat(opt.cache, 'trainCache_' .. opt.dataset .. '_' .. opt.frames .. '_' .. opt.fineWidth ..'_' .. opt.fineHeight .. '_' .. opt.flownet_factor .. '.t7')
local testCache = paths.concat(opt.cache, 'testCache_' .. opt.dataset .. '_' .. opt.frames .. '_' .. opt.fineWidth ..'_' .. opt.fineHeight .. '_' .. opt.flownet_factor .. '.t7')
if opt.original_pwc == 1 then
	trainCache = paths.concat(opt.cache, 'trainCache_' .. opt.dataset .. '_' .. opt.frames .. '_' .. opt.fineWidth ..'_' .. opt.fineHeight .. '_' .. opt.flownet_factor .. '_PWC_ORIGINAL.t7')
	testCache = paths.concat(opt.cache, 'testCache_' .. opt.dataset .. '_' .. opt.frames .. '_' .. opt.fineWidth ..'_' .. opt.fineHeight .. '_' .. opt.flownet_factor .. '_PWC_ORIGINAL.t7')
end

local meanstd = {
	mean = { 0.485, 0.456, 0.406 },
	std = { 0.229, 0.224, 0.225 },
}
local pca = {
	eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
	eigvec = torch.Tensor{
		{ -0.5675,  0.7192,  0.4009 },
		{ -0.5808, -0.0045, -0.8140 },
		{ -0.5836, -0.6948,  0.4203 },
	},
}

local mean = meanstd.mean
local std = meanstd.std
------------------------------------------
-- Warping Function:
local function createWarpModel()
	local imgData = nn.Identity()()
	local floData = nn.Identity()()

	local imgOut = nn.Transpose({2,3},{3,4})(imgData)
	local floOut = nn.Transpose({2,3},{3,4})(floData)

	local warpImOut = nn.Transpose({3,4},{2,3})(nn.BilinearSamplerBHWD()({imgOut, floOut}))
	local model = nn.gModule({imgData, floData}, {warpImOut})

	return model
end

local loadSize = opt.loadSize
local inputSize = {opt.channels, opt.fineHeight, opt.fineWidth}
local outputSize = {4, opt.fineHeight, opt.fineWidth}
local scaleFactor = opt.scale

local function getExamples(path)
	local samplefile = torch.DiskFile(path)
	local sampledata = samplefile:readString("*a")
	local examples = stringx.split(sampledata, "\n")
	samplefile:close()
	return examples
end
local examples = getExamples('datasets/' .. opt.dataset .. '.dat')

local function getTrainValidationSplits(path)
	local ff = torch.DiskFile(path, 'r')
	local numSamples = 0
	for _ in io.lines(path) do
		numSamples = numSamples + 1
	end
	local trainValidationSamples = torch.IntTensor(numSamples)
	ff:readInt(trainValidationSamples:storage())
	ff:close()

	local train_samples = trainValidationSamples:eq(1):nonzero()
	local val_samples = trainValidationSamples:eq(2):nonzero()

	return train_samples, val_samples
end
local train_samples, val_samples = getTrainValidationSplits('datasets/' .. opt.dataset .. '_split.dat')


local function loadImage(path)
	local input = image.load(path, 3, 'float')
	return input
end

local  function rotateFlow(flow, angle)
	local flow_rot = image.rotate(flow, angle)
	local fu = torch.mul(flow_rot[1], math.cos(-angle)) - torch.mul(flow_rot[2], math.sin(-angle))
	local fv = torch.mul(flow_rot[1], math.sin(-angle)) + torch.mul(flow_rot[2], math.cos(-angle))
	flow_rot[1]:copy(fu)
	flow_rot[2]:copy(fv)

	return flow_rot
end

local function scaleFlow(flow, height, width)
	-- scale the original flow to a flow of size height x width
	local sc = height/flow:size(2)
	assert(torch.abs(width/flow:size(3) - sc)<eps, 'Aspect ratio of output flow is not the same as input flow' )
	local flow_scaled = image.scale(flow, width, height)*sc
	return flow_scaled
end

local function makeData(images, flows, occs, mask, rand_crop)
	-- crop image if necessary
	local iW = images:size(3)
	local iH = images:size(2)
	local oW = inputSize[3]
	local oH = inputSize[2]
	local h1 = math.floor(torch.uniform(1e-2, iH-oH))
	local w1 = math.floor(torch.uniform(1e-2, iW-oW))

	-- scale
	if scaleFactor ~= 1 then
		local sc = '*' .. scaleFactor
		images = image.scale(images, sc)
		mask = image.scale(mask, sc)
		flows = scaleFlow(flows, iH * scaleFactor, iW * scaleFactor)
		occs = image.scale(occs, sc, 'simple')
	end

	local images_cropped, flows_cropped, occs_cropped, mask_cropped

	if rand_crop == 1 then 
		images_cropped = image.crop(images, w1, h1, w1 + oW, h1 + oH)
		mask_cropped = image.crop(mask, w1, h1, w1 + oW, h1 + oH)
		flows_cropped = image.crop(flows, w1, h1, w1 + oW, h1 + oH)
		occs_cropped = image.crop(occs, w1, h1, w1 + oW, h1 + oH)
	else
		images_cropped = image.crop(images, 'c', inputSize[3], inputSize[2])
		mask_cropped = image.crop(mask, 'c', inputSize[3], inputSize[2])
		flows_cropped = image.crop(flows, 'c', outputSize[3], outputSize[2])
		occs_cropped = image.crop(occs, 'c', outputSize[3], outputSize[2])
	end

	local output = torch.cat(flows_cropped, occs_cropped, 1)

	return images_cropped, output, mask_cropped
end


local function Preprocess()
	if opt.normalize_images == 1 then
		return TF.Compose{
			TF.ColorJitter({
				brightness = 0.02,
				contrast = 0.02,
				saturation = 0.02,
			}),
			TF.Lighting(0.1, pca.eigval, pca.eigvec),
			TF.ColorNormalize(meanstd),
		}
	else
		return TF.Compose{
			TF.ColorJitter({
				brightness = 0.02,
				contrast = 0.02,
				saturation = 0.02,
			}),
			TF.Lighting(0.1, pca.eigval, pca.eigvec),
		}
	end
end


-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(self, id)
	collectgarbage()
	local pathTable = stringx.split(examples[id], " ")

	local ref_idx = 2
	local skip_frame = 1
	if opt.ground_truth == true then
		ref_idx = 3
		if #pathTable == 4 then
			skip_frame = pathTable[4]
		end
	else
		if #pathTable == 3 then
			skip_frame = pathTable[3]
		end
	end

	local ref = pathTable[ref_idx]
	local s = ref
	if opt.frames > 2 then
		s = s - 0.5 * (opt.frames - 1) * skip_frame
	end

	local img_array = {}
	local all_ref = 1
	local all_win = 1
	if opt.frames > 2 then
		all_ref = 0.5 * (opt.frames + 1)
		all_win = all_ref - 1
	end

	for f = 1,opt.frames do
		local img = loadImage(string.format(pathTable[1], s + (f - 1) * skip_frame))
		img_array[f] = img
	end

	local flow, occ, mask
	if opt.ground_truth == true then
		local pathF = string.format(pathTable[2], ref)
		
		flow, mask = flowX.loadFlow(pathF)

		local pathD
		if opt.frames == 2 then
			pathD = string.gsub(pathF, ".flo", string.format("_occ_%i.disp", 3))
		else
			pathD = string.gsub(pathF, ".flo", string.format("_occ_%i.disp", opt.frames))
		end
		local f = io.open(pathD,"r")
		if f~=nil then
			io.close(f)
			occ = flowX.loadDISP(pathD)
      		occ = occ:view(1, occ:size(1), occ:size(2))
		else
			occ = torch.Tensor(1,flow:size(2), flow:size(3)):fill(0.5)
		end

		pathD = string.gsub(pathF, ".flo", string.format("_occ_%i.disp", 3))
		local f = io.open(pathD,"r")
		if f~=nil then
			io.close(f)
			local tmp = flowX.loadDISP(pathD)
			tmp = tmp:view(1, tmp:size(1), tmp:size(2))
			occ = torch.cat(occ, tmp, 1)
		else
			local tmp = torch.Tensor(1,flow:size(2), flow:size(3)):fill(0.5)
			occ = torch.cat(occ, tmp, 1)
		end
	else
		flow = torch.Tensor(2, img_array[1]:size(2), img_array[1]:size(3)):zero()
		occ = torch.Tensor(2,flow:size(2), flow:size(3)):fill(0.5)
	end
  if not mask then
    mask = torch.FloatTensor(1, flow:size(2), flow:size(3)):fill(1)
  end

	if opt.gaussian_noise > 0 then
		for f = 1,opt.frames do
			-- Add Random Noise to the images
			img_array[f] = img_array[f]:add(torch.randn(img_array[f]:size()) * opt.gaussian_noise)
			local mask = torch.ge(img_array[f], 0):cmul(torch.le(img_array[f], 1)):float() -- if in [0,1] 1 otherwise 0
			img_array[f]:cmul(mask)
		end
	end

	local images
	if opt.augment == 1 then
		local iW = img_array[1]:size(3)
		local iH = img_array[1]:size(2)
		local oW = loadSize[3]
		local oH = loadSize[2]

		-- do hflip and vflip with probability 0.5
		if torch.uniform() > 0.5 then
			for f = 1,opt.frames do
				img_array[f] = image.hflip(img_array[f])
			end
			flow = image.hflip(flow)
			flow[1] = flow[1]*(-1)
			occ = image.hflip(occ)
		end
		if torch.uniform() > 0.5 then
			for f = 1,opt.frames do
				img_array[f] = image.vflip(img_array[f])
			end
			flow = image.vflip(flow)
			flow[2] = flow[2]*(-1)
			occ = image.vflip(occ)
		end

		--apply data augmentation : random translation and rotation
		local t = 10*torch.rand(2)
		local r1,r2 = torch.uniform(-0.2,0.2),torch.uniform(-0.1,0.1)

		--generate flowamp from rotation between the 2 frames
		local rotate_flow = torch.Tensor():resizeAs(flow)
		for i=1,iW do
			rotate_flow[2][{{},i}]:fill((i-iW/2)*(-r2))
		end
		for i=1,iH do
			rotate_flow[1][i]:fill((i-iH/2)*(r2))
		end

		--data augmentation
		flow:add(rotate_flow)

		flow = image.rotate(flow,r1)
		--rotate flow vectors
		local flow_ = flow:clone()
		flow[1] = math.cos(r1)*flow_[1] + math.sin(r1)*flow_[2]
		flow[2] = -math.sin(r1)*flow_[1] + math.cos(r1)*flow_[2]

		img_array[all_ref] = image.rotate(img_array[all_ref],r1)
		mask = image.rotate(mask,r1)
		for f = 1,all_win do
			if opt.frames > 2 then
				img_array[all_ref - f] = image.rotate(img_array[all_ref - f],r1 - (f * r2))
				img_array[all_ref - f] = image.translate(img_array[all_ref - f],-f * t[1],-f * t[2])
			end

			img_array[all_ref + f] = image.rotate(img_array[all_ref + f],r1 + (f * r2))
			img_array[all_ref + f] = image.translate(img_array[all_ref + f],f * t[1],f * t[2])
		end

		flow[1] = flow[1] + t[1]
		flow[2] = flow[2] + t[2]

		-- concat the images to one input
		for f = 1,opt.frames do
			if f == 1 then
				images = img_array[f]
			else
				images = torch.cat(images, img_array[f], 1)
			end
		end

		-- Add Random Scale
		local sc = torch.uniform(1.0, 2.0)

		images = image.scale(images, '*'..sc)
		mask = image.scale(mask, '*'..sc)
		occ = image.scale(occ, '*'..sc, 'simple')
		flow = image.scale(flow, '*'..sc)*sc   -- Notice the scaling of flow here

		iW = images:size(3)
		iH = images:size(2)
		local h1 = math.floor(torch.uniform(1, iH-oH))
		local w1 = math.floor(torch.uniform(1, iW-oW))
		imagesOut = image.crop(images, w1, h1, w1 + oW, h1 + oH)
		maskOut = image.crop(mask, w1, h1, w1 + oW, h1 + oH)
		flowOut = image.crop(flow, w1, h1, w1 + oW, h1 + oH)
		occOut = image.crop(occ, w1, h1, w1 + oW, h1 + oH)

		imagesOut = Preprocess()(imagesOut)

		assert(imagesOut:size(3) == oW)
		assert(imagesOut:size(2) == oH)
		assert(maskOut:size(3) == oW)
		assert(maskOut:size(2) == oH)
		assert(flowOut:size(3) == oW)
		assert(flowOut:size(2) == oH)
		assert(occOut:size(3) == oW)
		assert(occOut:size(2) == oH)
	else
		-- concat the images to one input
		for f = 1,opt.frames do
			if f == 1 then
				images = img_array[f]
			else
				images = torch.cat(images, img_array[f], 1)
			end
		end

		if opt.normalize_images == 1 then
			imagesOut = TF.ColorNormalize(meanstd)(images)
		else
			imagesOut = images
		end
		maskOut = mask
		flowOut = flow
		occOut = occ
	end


	if opt.flownet_factor ~= 1 then
		flowOut:div(opt.flownet_factor)
	end

	return makeData(imagesOut, flowOut, occOut, maskOut, opt.rand_crop)
end

-- split into training and validation set
local train_split, val_split
if val_samples:nElement() > 0 then
	train_split = math.floor((100 * train_samples:size(1)) / (train_samples:size(1) + val_samples:size(1)))
	val_split = math.floor((100 * val_samples:size(1)) / (train_samples:size(1) + val_samples:size(1)))
else
	train_split = 100
	val_split = 0
end

print("training ")
if paths.filep(trainCache) then
	print('Loading train metadata from cache')
	trainLoader = torch.load(trainCache)
	trainLoader.sampleHookTrain = trainHook
	print(trainLoader.samplingIds:size(1))
else
	print('Creating train metadata')
	trainLoader = dataLoader{
		loadSize = loadSize,
		inputSize = inputSize,
		outputSize = outputSize,
		split = train_split,
		--split = 100,
		samplingIds = train_samples,
		verbose = true
	}
	torch.save(trainCache, trainLoader)
	trainLoader.sampleHookTrain = trainHook
end
collectgarbage()

-- End of train loader section
--------------------------------------------------------------------------------
--[[
   Section 2: Create a test data loader (testLoader),
   which can iterate over the test set and returns an image's
--]]

local testHook = function(self, id)
	collectgarbage()
	local pathTable = stringx.split(examples[id], " ")

	local ref_idx = 2
	local skip_frame = 1
	if opt.ground_truth == true then
		ref_idx = 3
		if #pathTable == 4 then
			skip_frame = pathTable[4]
		end
	else
		if #pathTable == 3 then
			skip_frame = pathTable[3]
		end
	end

	local ref = pathTable[ref_idx]
	local s = ref
	if opt.frames > 2 then
		s = s - 0.5 * (opt.frames - 1) * skip_frame
	end

	local images
	for f = 1,opt.frames do
		local img = loadImage(string.format(pathTable[1], s + (f - 1) * skip_frame))
		if f == 1 then
			images = img
		else
			images = torch.cat(images, img, 1)
		end
	end

	local flow, occ, mask
	if opt.ground_truth == true then
		local pathF = string.format(pathTable[2], ref)
		
		flow, mask = flowX.loadFlow(pathF)

		if opt.flownet_factor ~= 1 then
			flow:div(opt.flownet_factor)
		end

		local pathD
		if opt.frames == 2 then
			pathD = string.gsub(pathF, ".flo", string.format("_occ_%i.disp", 3))
		else
			pathD = string.gsub(pathF, ".flo", string.format("_occ_%i.disp", opt.frames))
		end
		local f = io.open(pathD,"r")
		if f~=nil then
			io.close(f)
			occ = flowX.loadDISP(pathD)
			occ = occ:view(1, occ:size(1), occ:size(2))
		else
			occ = torch.Tensor(1, flow:size(2), flow:size(3)):fill(0.5)
		end

		pathD = string.gsub(pathF, ".flo", string.format("_occ_%i.disp", 3))
		local f = io.open(pathD,"r")
		if f~=nil then
		io.close(f)
		local tmp = flowX.loadDISP(pathD)
		tmp = tmp:view(1, tmp:size(1), tmp:size(2))
		occ = torch.cat(occ, tmp, 1)
		else
		local tmp = torch.Tensor(1, flow:size(2), flow:size(3)):fill(0.5)
		occ = torch.cat(occ, tmp, 1)
		end
	else
		flow = torch.Tensor(2, images:size(2), images:size(3)):zero()
		occ = torch.Tensor(2, flow:size(2), flow:size(3)):fill(0.5)
	end
	if not mask then
		mask = torch.FloatTensor(1, flow:size(2), flow:size(3)):fill(1)
	end

	images = TF.ColorNormalize(meanstd)(images)

	return makeData(images, flow, occ, mask, 0)
end

print("testing ")
if paths.filep(testCache) then
	print('Loading test metadata from cache')
	testLoader = torch.load(testCache)
	testLoader.sampleHookTest = testHook
	print(testLoader.samplingIds:size(1))
else
	print('Creating test metadata')
	testLoader = dataLoader{
		loadSize = loadSize,
		inputSize = inputSize,
		outputSize = outputSize,
		split = val_split,
		samplingIds = val_samples,
		verbose = true
	}
	torch.save(testCache, testLoader)
	testLoader.sampleHookTest = testHook
end
collectgarbage()
-- End of test loader section
