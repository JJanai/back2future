----------------------------------------------------
---- L1 PENALTY FUNCTION
-----------------------------------------------------
-- Robust modified L1 penalty function for losses.
--
-- Copyright 2018 Joel Janai, Fatma Güney, Anurag Ranjan and the Max Planck Gesellschaft.
-- All rights reserved.
-- This software is provided for research purposes only.
-- By using this software you agree to the terms of the license file
-- in the root folder.
---------------------------------------------------------------

local L1Penalty = torch.class('L1Penalty')

function L1Penalty:__init(alpha)
  self.eps = 0.001 * 0.001
  self.alpha = 0.5 or alpha
end

function L1Penalty:apply(x)
  return torch.pow(x, 2):add(self.eps):pow(self.alpha)
end

function L1Penalty:der(x)
  return torch.mul(x, 2*self.alpha):cdiv(torch.pow(x, 2):add(self.eps):pow(1-self.alpha))
end
