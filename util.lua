-----------------------------------------------------
-- Some utility functions
--
-- Copyright 2018 Joel Janai, Fatma Güney, Anurag Ranjan and the Max Planck Gesellschaft.
-- All rights reserved.
-- This software is provided for research purposes only.
-- By using this software you agree to the terms of the license file
-- in the root folder.
---------------------------------------------------------------

require 'cunn'
require 'nngraph'
require 'stn'
require 'spy'
local ffi=require 'ffi'

function file_exists(name)
  local f = io.open(name,"r")
  if f~=nil then 
    io.close(f) 
    return true 
  else 
    return false 
  end
end

function makeDataParallel(model, nGPU)
    if nGPU > 1 then
        local gpus = torch.range(1, nGPU):totable()
        local fastest, benchmark = cudnn.fastest, cudnn.benchmark
 
        local dpt = nn.DataParallelTable(1, true, true)
             :add(model, gpus)
             :threads(function()
                 local cudnn = require 'cudnn'
                 local nngraph = require 'nngraph'
                 local stn = require 'stn'
                 paths.dofile('models/CostVolMulti.lua')
                 cudnn.fastest, cudnn.benchmark = fastest, benchmark
             end)
        dpt.flow_scale = model.flow_scale
        dpt.amplify = model.amplify
        
        dpt.gradInput = nil
        model = dpt:cuda()
    end
    return model
end

local function cleanDPT(module)
   -- This assumes this DPT was created by the function above: all the
   -- module.modules are clones of the same network on different GPUs
   -- hence we only need to keep one when saving the model to the disk.
   local newDPT = nn.DataParallelTable(1)
   cutorch.setDevice(opt.GPU)
   newDPT:add(module:get(1), opt.GPU)
   return newDPT
end

function saveDataParallel(filename, model)
   if torch.type(model) == 'nn.DataParallelTable' then
      torch.save(filename, cleanDPT(model))
   elseif torch.type(model) == 'nn.Sequential' then
      local temp_model = nn.Sequential()
      for i, module in ipairs(model.modules) do
         if torch.type(module) == 'nn.DataParallelTable' then
            temp_model:add(cleanDPT(module))
         else
            temp_model:add(module)
         end
      end
      torch.save(filename, temp_model)
   elseif torch.type(model) == 'nn.gModule' then
      torch.save(filename, model)
   else
      error('This saving function only works with Sequential or DataParallelTable modules.')
   end
end

function loadDataParallel(filename, nGPU)
   if opt.backend == 'cudnn' then
      require 'cudnn'
   end
   local model = torch.load(filename)
   if torch.type(model) == 'nn.DataParallelTable' then
      return makeDataParallel(model:get(1), nGPU)
   elseif torch.type(model) == 'nn.Sequential' then
      for i,module in ipairs(model.modules) do
         if torch.type(module) == 'nn.DataParallelTable' then
            model.modules[i] = makeDataParallel(module:get(1):float(), nGPU)
         end
      end
      return model
   elseif torch.type(model) == 'nn.gModule' then
      return model
   else
      error('The loaded model is not a Sequential or DataParallelTable module.')
   end
end

--string utils
function starts_with(str, start)
  return string.sub(str,1,string.len(start)) == start
end

function ends_with(str, back)
  local e = back:len()
  return string.sub(str, -e) == back;
end

function split(str, sep)
   local sep, fields = sep or ":", {}
   local pattern = string.format("([^%s]+)", sep)
   str:gsub(pattern, function(c) fields[#fields+1] = c end)
   return fields
end

function basename(filename)
  local idx = filename:match(".+()%.%w+$")
  if(idx) then
    return filename:sub(1, idx-1)
  else
    return filename
  end
end

function getLatestModelSaved(saveDir)
  local latest = -1
  for file in paths.iterfiles(saveDir) do
    local s = split(basename(file), '_')
    if #s == 2 and s[1] == 'model' then
      local model_n = tonumber(s[2])
      if model_n > latest then
        latest = model_n
      end
    end
  end
  
  return latest
end
