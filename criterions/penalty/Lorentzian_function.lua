----------------------------------------------------
---- LORENTZIAN PENALTY FUNCTION
-----------------------------------------------------
-- Robust lorentzian penalty function for losses.
--
-- Copyright 2018 Joel Janai, Fatma Güney, Anurag Ranjan and the Max Planck Gesellschaft.
-- All rights reserved.
-- This software is provided for research purposes only.
-- By using this software you agree to the terms of the license file
-- in the root folder.
---------------------------------------------------------------

local Lorentzian = torch.class('LorentzianPenalty')

function Lorentzian:__init() 
  self.eps = 0.05
  self.eps_sq = self.eps * self.eps
end

function Lorentzian:set_eps(eps)
  self.eps = eps
  self.eps_sq = self.eps * self.eps
end

function Lorentzian:apply(x) 
  return torch.log(1 + 0.5 * torch.div(torch.pow(x, 2), self.eps_sq));
end

function Lorentzian:der(x) 
  return torch.cdiv(torch.mul(x,2), torch.add(torch.pow(x, 2), 2 * self.eps_sq))
end