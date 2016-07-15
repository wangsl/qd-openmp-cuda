
function [ P ] = AssLegendreP(Omega, LMax, x)

assert(LMax >= Omega)

L = LMax - Omega + 1;
P = zeros(numel(x), L);

for l = Omega : LMax
  p = legendre(l, x, 'norm');
  P(:, l-Omega+1) = p(Omega+1, :)';
end

return
