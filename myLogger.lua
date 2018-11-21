--[[
This code is an adaption from https://github.com/torch/optim/blob/master/Logger.lua

myLogger: a simple class to log symbols during training, and automate plot generation

Example:
    myLogger = optim.myLogger('somefile.log')    -- file to save stuff

    for i = 1,N do                           -- log some symbols during
        train_error = ...                     -- training/testing
        test_error = ...
        myLogger:add{['training error'] = train_error,
            ['test error'] = test_error}
    end

    myLogger:style{['training error'] = '-',   -- define styles for plots
                 ['test error'] = '-'}
    myLogger:plot()                            -- and plot

---- OR ---

    myLogger = optim.myLogger('somefile.log')    -- file to save stuff
    myLogger:setNames{'training error', 'test error'}

    for i = 1,N do                           -- log some symbols during
       train_error = ...                     -- training/testing
       test_error = ...
       myLogger:add{train_error, test_error}
    end

    myLogger:style{'-', '-'}                   -- define styles for plots
    myLogger:plot()                            -- and plot

-----------

    myLogger:setlogscale(true)                 -- enable logscale on Y-axis
    myLogger:plot()                            -- and plot
]]
require 'xlua'
local myLogger = torch.class('optim.myLogger')

function myLogger:__init(filename, timestamp)
   if filename then
      self.name = filename
      os.execute('mkdir ' .. (sys.uname() ~= 'windows' and '-p ' or '') .. ' "' .. paths.dirname(filename) .. '"')
      if timestamp then
         -- append timestamp to create unique log file
         filename = filename .. '-'..os.date("%Y_%m_%d_%X")
      end
      self.file = io.open(filename,'a')
      self.epsfile = self.name .. '.eps'
   else
      self.file = io.stdout
      self.name = 'stdout'
      print('<myLogger> warning: no path provided, logging to std out')
   end
   self.empty = true
   self.symbols = {}
   self.styles = {}
   self.names = {}
   self.idx = {}
   self.figure = nil
   self.showPlot = true
   self.plotRawCmd = nil
   self.defaultStyle = '+'
   self.logscale = false
end

function myLogger:setNames(names)
   self.names = names
   self.empty = false
   self.nsymbols = #names
   for k,key in pairs(names) do
      self.file:write(key .. '\t')
      self.symbols[k] = {}
      self.styles[k] = {self.defaultStyle}
      self.idx[key] = k
   end
   self.file:write('\n')
   self.file:flush()
   return self
end

function myLogger:add(symbols)
   -- (1) first time ? print symbols' names on first row
   if self.empty then
      self.empty = false
      self.nsymbols = #symbols
      for k,val in pairs(symbols) do
         self.file:write(k .. '\t')
         self.symbols[k] = {}
         self.styles[k] = {self.defaultStyle}
         self.names[k] = k
      end
      self.idx = self.names
      self.file:write('\n')
   end
   -- (2) print all symbols on one row
   for k,val in pairs(symbols) do
      if type(val) == 'number' then
         self.file:write(string.format('%11.4e',val) .. '\t')
      elseif type(val) == 'string' then
         self.file:write(val .. '\t')
      else
         xlua.error('can only log numbers and strings', 'myLogger')
      end
   end
   self.file:write('\n')
   self.file:flush()
   -- (3) save symbols in internal table
   for k,val in pairs(symbols) do
      table.insert(self.symbols[k], val)
   end
end

function myLogger:style(symbols)
   for name,style in pairs(symbols) do
      if type(style) == 'string' then
         self.styles[name] = {style}
      elseif type(style) == 'table' then
         self.styles[name] = style
      else
         xlua.error('style should be a string or a table of strings','myLogger')
      end
   end
   return self
end

function myLogger:setlogscale(state)
   self.logscale = state
end

function myLogger:display(state)
   self.showPlot = state
end

function myLogger:plot(...)
   if not xlua.require('gnuplot') then
      if not self.warned then
         print('<myLogger> warning: cannot plot with this version of Torch')
         self.warned = true
      end
      return
   end
   local plotit = false
   local plots = {}
   local plotsymbol =
      function(name,list)
         if #list > 1 then
            local nelts = #list
            local plot_y = torch.Tensor(nelts)
            for i = 1,nelts do
               plot_y[i] = list[i]
            end
            for _,style in ipairs(self.styles[name]) do
               table.insert(plots, {self.names[name], plot_y, style})
            end
            plotit = true
         end
      end
   local args = {...}
   if not args[1] then -- plot all symbols
      for name,list in pairs(self.symbols) do
         plotsymbol(name,list)
      end
   else -- plot given symbols
      for _,name in ipairs(args) do
         plotsymbol(self.idx[name], self.symbols[self.idx[name]])
      end
   end
   if plotit then
      if self.showPlot then
         self.figure = gnuplot.figure(self.figure)
         if self.logscale then gnuplot.logscale('on') end
         gnuplot.plot(plots)
         if self.plotRawCmd then gnuplot.raw(self.plotRawCmd) end
         gnuplot.grid('on')
         gnuplot.title('<myLogger::' .. self.name .. '>')
      end
      if self.epsfile then
         os.execute('rm -f "' .. self.epsfile .. '"')
         local epsfig = gnuplot.epsfigure(self.epsfile)
         if self.logscale then gnuplot.logscale('on') end
         gnuplot.plot(plots)
         if self.plotRawCmd then gnuplot.raw(self.plotRawCmd) end
         gnuplot.grid('on')
         gnuplot.title('<myLogger::' .. self.name .. '>')
         gnuplot.plotflush()
         gnuplot.close(epsfig)
      end
   end
end
