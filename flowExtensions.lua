-- Copyright 2018 Joel Janai, Fatma Güney, Anurag Ranjan and the Max Planck Gesellschaft.
-- All rights reserved.
-- This software is provided for research purposes only.  
-- By using this software you agree to the terms of the license file 
-- in the root folder.
-- For commercial use, please contact ps-license@tue.mpg.de.
require 'image'

-------------------------
-- Optical Flow Utilities
-------------------------
local stringx = require('pl.stringx')
local M = {}

local eps = 1e-6

local function computeNorm(...)
   -- check args
   local _, flow_x, flow_y = xlua.unpack(
      {...},
      'opticalflow.computeNorm',
      'computes norm (size) of flow field from flow_x and flow_y,\n',
      {arg='flow_x', type='torch.Tensor', help='flow field (x), (WxH)', req=true},
      {arg='flow_y', type='torch.Tensor', help='flow field (y), (WxH)', req=true}
   )
   local flow_norm = torch.Tensor()
   local x_squared = torch.Tensor():resizeAs(flow_x):copy(flow_x):cmul(flow_x)
   flow_norm:resizeAs(flow_y):copy(flow_y):cmul(flow_y):add(x_squared):sqrt()
   return flow_norm
end
M.computeNorm = computeNorm

------------------------------------------------------------
-- computes angle (direction) of flow field from flow_x and flow_y,
--
-- @usage opticalflow.computeAngle() -- prints online help
--
-- @param flow_x  flow field (x), (WxH) [required] [type = torch.Tensor]
-- @param flow_y  flow field (y), (WxH) [required] [type = torch.Tensor]
------------------------------------------------------------
local function computeAngle(...)
   -- check args
   local _, flow_x, flow_y = xlua.unpack(
      {...},
      'opticalflow.computeAngle',
      'computes angle (direction) of flow field from flow_x and flow_y,\n',
      {arg='flow_x', type='torch.Tensor', help='flow field (x), (WxH)', req=true},
      {arg='flow_y', type='torch.Tensor', help='flow field (y), (WxH)', req=true}
   )
   local flow_angle = torch.Tensor()
   flow_angle:resizeAs(flow_y):copy(flow_y):cdiv(flow_x):abs():atan():mul(180/math.pi)
   flow_angle:map2(flow_x, flow_y, function(h,x,y)
              if x == 0 and y >= 0 then
           return 90
              elseif x == 0 and y <= 0 then
           return 270
              elseif x >= 0 and y >= 0 then
           -- all good
              elseif x >= 0 and y < 0 then
           return 360 - h
              elseif x < 0 and y >= 0 then
           return 180 - h
              elseif x < 0 and y < 0 then
           return 180 + h
              end
           end)
   return flow_angle
end
M.computeAngle = computeAngle
------------------------------------------------------------
-- merges Norm and Angle flow fields into a single RGB image,
-- where saturation=intensity, and hue=direction
--
-- @usage opticalflow.field2rgb() -- prints online help
--
-- @param norm  flow field (norm), (WxH) [required] [type = torch.Tensor]
-- @param angle  flow field (angle), (WxH) [required] [type = torch.Tensor]
-- @param max  if not provided, norm:max() is used [type = number]
-- @param legend  prints a legend on the image [type = boolean]
------------------------------------------------------------
local function field2rgb(...)
   -- check args
   local _, norm, angle, max, legend = xlua.unpack(
      {...},
      'opticalflow.field2rgb',
      'merges Norm and Angle flow fields into a single RGB image,\n'
   .. 'where saturation=intensity, and hue=direction',
      {arg='norm', type='torch.Tensor', help='flow field (norm), (WxH)', req=true},
      {arg='angle', type='torch.Tensor', help='flow field (angle), (WxH)', req=true},
      {arg='max', type='number', help='if not provided, norm:max() is used'},
      {arg='legend', type='boolean', help='prints a legend on the image', default=false}
   )
   
   -- max
   local saturate = false
   if max then saturate = true end
   max = math.max(max or norm:max(), 1e-2)
   
   -- merge them into an HSL image
   local hsl = torch.Tensor(3,norm:size(1), norm:size(2))
   -- hue = angle:
   hsl:select(1,1):copy(angle):div(360)
   -- saturation = normalized intensity:
   hsl:select(1,2):copy(norm):div(max)
   if saturate then hsl:select(1,2):tanh() end
   -- light varies inversely from saturation (null flow = white):
   hsl:select(1,3):copy(hsl:select(1,2)):mul(-0.5):add(1)
   
   -- convert HSL to RGB
   local rgb = image.hsl2rgb(hsl)
   
   -- legend
   if legend then
      _legend_ = _legend_
   or image.load(paths.concat(paths.install_lua_path, 'opticalflow/legend.png'),3)
      legend = torch.Tensor(3,hsl:size(2)/8, hsl:size(2)/8)
      image.scale(_legend_, legend, 'bilinear')
      rgb:narrow(1,1,legend:size(2)):narrow(2,hsl:size(2)-legend:size(2)+1,legend:size(2)):copy(legend)
   end
   
   -- done
   return rgb, max
end
M.field2rgb = field2rgb
------------------------------------------------------------
-- Simplifies display of flow field in HSV colorspace when the
-- available field is in x,y displacement
--
-- @usage opticalflow.xy2rgb() -- prints online help
--
-- @param x  flow field (x), (WxH) [required] [type = torch.Tensor]
-- @param y  flow field (y), (WxH) [required] [type = torch.Tensor]
------------------------------------------------------------
local function xy2rgb(...)
   -- check args
   local _, x, y, max = xlua.unpack(
      {...},
      'opticalflow.xy2rgb',
      'merges x and y flow fields into a single RGB image,\n'
   .. 'where saturation=intensity, and hue=direction',
      {arg='x', type='torch.Tensor', help='flow field (norm), (WxH)', req=true},
      {arg='y', type='torch.Tensor', help='flow field (angle), (WxH)', req=true},
      {arg='max', type='number', help='if not provided, norm:max() is used'}
   )
   
   local norm = computeNorm(x,y)
   local angle = computeAngle(x,y)
   return field2rgb(norm,angle,max)
end
M.xy2rgb = xy2rgb

local function loadDISP(filename)
  TAG_FLOAT = 202021.25 
  local ff = torch.DiskFile(filename):binary()
  local tag = ff:readFloat()
  if tag ~= TAG_FLOAT then
    xerror('unable to read '..filename..
     ' perhaps bigendian error','readflo()')
  end
   
  local w = ff:readInt()
  local h = ff:readInt()
  local tf = torch.FloatTensor(h, w)
  ff:readFloat(tf:storage())
  ff:close()

  local occ = tf:permute(1,2)  
  return occ
end
M.loadDISP = loadDISP

local function getOCC(depth, flow, frames)
  local h = depth:size(1)
  local w = depth:size(2)
  -- get occlusion from depth
  local fwd_pixel = torch.Tensor(h,w):fill(-1)
  local fwd_warped = torch.Tensor(h,w):zero()
  local bwd_pixel = torch.Tensor(h,w):fill(-1)
  local bwd_warped = torch.Tensor(h,w):zero()
  local occlusion = torch.Tensor(h,w):fill(0.5)
  for x=1,w do
    for y=1,h do
      local i = (x-1) * h + (y-1)
      local xf = torch.round(x + flow[1][y][x])
      local yf = torch.round(y + flow[2][y][x]) 
      
      if xf > 0 and xf <= w and yf > 0 and yf <= h then 
        if fwd_pixel[yf][xf] == -1 then
          fwd_pixel[yf][xf] = i
          fwd_warped[yf][xf] = depth[y][x]
        elseif depth[y][x] - fwd_warped[yf][xf] < -0.1 then
          local occ_x = math.floor(fwd_pixel[yf][xf] / h) + 1
          local occ_y = fwd_pixel[yf][xf] % h + 1 
          
          occlusion[occ_y][occ_x] = 1
          
          fwd_pixel[yf][xf] = i
          fwd_warped[yf][xf] = depth[y][x]
        else
          occlusion[y][x] = 1
        end
      else
        occlusion[y][x] = 1
      end
        
      -- past
      xf = torch.round(x - flow[1][y][x])
      yf = torch.round(y - flow[2][y][x]) 
      if xf > 0 and xf <= w and yf > 0 and yf <= h then 
        if bwd_pixel[yf][xf] == -1 then
          bwd_pixel[yf][xf] = i
          bwd_warped[yf][xf] = depth[y][x]
        elseif depth[y][x] - bwd_warped[yf][xf] < -0.1 then
          local occ_x = math.floor(bwd_pixel[yf][xf] / h) + 1
          local occ_y = bwd_pixel[yf][xf] % h + 1 
          
          occlusion[occ_y][occ_x] = 0
          
          bwd_pixel[yf][xf] = i
          bwd_warped[yf][xf] = depth[y][x]
        else
          occlusion[y][x] = 0
        end
      else
        occlusion[y][x] = 0
      end
    end
  end
  
  local tmp = occlusion
  for x=1,w do
    for y=1,h do
      local n = tmp[{{math.max(y-1,1),math.min(y+1,h)},{math.max(x-1,1),math.min(x+1,w)}}]:clone()
      occlusion[y][x] = n:view(n:nElement()):median()
    end
  end
  return occlusion
end
M.getOCC = getOCC

local function loadFlow(pathF) 
  if pathF:sub(pathF:len()-3,pathF:len()) == '.flo' then
    return M.loadFLO(pathF)
  elseif pathF:sub(pathF:len()-3,pathF:len()) == '.pfm' then
    return M.loadPFM(pathF)
  elseif pathF:sub(pathF:len()-3,pathF:len()) == '.png' then
    return M.loadKittiPNG(pathF)
  else
    error("Unknown flow format!")
  end
end
M.loadFlow = loadFlow

local function loadFLO(filename)
  TAG_FLOAT = 202021.25 
  local ff = torch.DiskFile(filename):binary()
  local tag = ff:readFloat()
  if tag ~= TAG_FLOAT then
    xerror('unable to read '..filename..
     ' perhaps bigendian error','readflo()')
  end
   
  local w = ff:readInt()
  local h = ff:readInt()
  local nbands = 2
  local tf = torch.FloatTensor(h, w, nbands)
  ff:readFloat(tf:storage())
  ff:close()

  local flow = tf:permute(3,1,2)  
  return flow
end
M.loadFLO = loadFLO

local function writeFLO(filename, F)
  F = F:permute(2,3,1):clone()
  TAG_FLOAT = 202021.25 
  local ff = torch.DiskFile(filename, 'w'):binary()
  ff:writeFloat(TAG_FLOAT)
   
  ff:writeInt(F:size(2)) -- width
  ff:writeInt(F:size(1)) -- height

  ff:writeFloat(F:storage())
  ff:close()
end
M.writeFLO = writeFLO

local function loadPFM(filename)
    ff = torch.DiskFile(filename):binary()
    local header = ff:readString("*l")
    local color, nbands
    if stringx.strip(header) == 'PF' then
        color = true
        nbands = 3
    else
        color = false
        nbands = 1
    end
    local dims = stringx.split(ff:readString("*l"))
    local scale = ff:readString("*l")
    if tonumber(scale) < 0 then
        ff:littleEndianEncoding()
    else
        ff:bigEndianEncoding()
    end
    local tf = ff:readFloat(dims[1]*dims[2]*nbands)    
    ff:close()
    tf = torch.FloatTensor(tf):resize(dims[2],dims[1],nbands):permute(3,1,2)
    tf = image.vflip(tf)
    return tf[{{1,2},{},{}}]
end
M.loadPFM = loadPFM

local function loadKittiPNG(filename)
  local img = image.loadPNG(filename, 16)
  img = img:float()
  
  img[{{1,2},{},{}}] = (img[{{1,2},{},{}}] * 65535 - math.pow(2,15)) / 64.0
  
  return img[{{1,2},{},{}}], img[{{3},{},{}}]
end
M.loadKittiPNG = loadKittiPNG

local function savePNG(filename, tensor)
   if not xlua.require 'liblua_png' then
      dok.error('libpng package not found, please install libpng','image.savePNG')
   end
   local save_to_file = 1
   tensor.libpng.save(filename, tensor, save_to_file)
end
M.savePNG = savePNG

local function rotate(flow, angle)
  local flow_rot = image.rotate(flow, angle, 'simple')
  local fu = torch.mul(flow_rot[1], math.cos(-angle)) - torch.mul(flow_rot[2], math.sin(-angle)) 
  local fv = torch.mul(flow_rot[1], math.sin(-angle)) + torch.mul(flow_rot[2], math.cos(-angle))
  flow_rot[1]:copy(fu)
  flow_rot[2]:copy(fv)

  return flow_rot
end
M.rotate = rotate

local function scale(flow, sc, opt)
  opt = opt or 'simple'
  local flow_scaled = image.scale(flow, '*'..sc, opt)*sc

  return flow_scaled

end
M.scale = scale

local function scaleBatch(flow, sc)
  local flowR = torch.FloatTensor(opt.batchSize*2, flow:size(3), flow:size(4))
  local outputR = torch.FloatTensor(opt.batchSize, 2, flow:size(3)*sc, flow:size(4)*sc)
  
  flowR:copy(flow)
  local output = image.scale(flowR, '*'..sc, 'simple')*sc
  outputR:copy(output)
  return outputR
end
M.scaleBatch = scaleBatch

return M
