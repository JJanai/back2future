----------------------------------------------------
---- QUADRATIC PENALTY FUNCTION
-----------------------------------------------------
-- Quadratic penalty function for losses.
--
-- Copyright 2018 Joel Janai, Fatma Güney, Anurag Ranjan and the Max Planck Gesellschaft.
-- All rights reserved.
-- This software is provided for research purposes only.
-- By using this software you agree to the terms of the license file
-- in the root folder.
---------------------------------------------------------------

local QuadraticPenalty = torch.class('QuadraticPenalty')

function QuadraticPenalty:apply(x) 
  return torch.pow(x,2)
end

function QuadraticPenalty:der(x) 
  return torch.mul(x,2)
end