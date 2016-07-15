
function [ Omega ] = OmegaList(J, p, LMax)

OmegaMax = min(J, LMax);

if J == 0
  OmegaMin = 0;
  OmegaMax = 0;
  p = 0;
elseif rem(J+p, 2) == 0
  OmegaMin = 0;
else
  OmegaMin = 1;
end

Omega = OmegaMin:OmegaMax;
