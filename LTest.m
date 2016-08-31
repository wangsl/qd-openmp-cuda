
%close all
%clear all
%clc

%format short;

global FH2Data

%FH2main

%L = FH2Data.theta.associated_legendre';
P = FH2Data.OmegaStates.associated_legendres;

[ m, LMax ] = size(P);

%for l = 1 : LMax
%  L(:,l) = sqrt(l-0.5)*L(:,l);
%end

% max(max(abs(L-P)))

psi = FH2Data.psi;

[ n1, n2, ~ ] = size(psi);
psi = reshape(psi, n1*n2, m);

psi = psi(1:2:end, :) + j*psi(2:2:end, :);

w = FH2Data.theta.w';
dr1 = FH2Data.r1.dr;
dr2 = FH2Data.r2.dr;

sum(w.*sum(abs(psi).^2, 1))*dr1*dr2

wP = zeros(m, LMax);
for k = 1 : m
  wP(k, :) = w(k).*P(k,:);
end

%%% Legendre Transform test

for i = 1 : 10
  s=sum(w.*sum(abs(psi).^2, 1))*dr1*dr2;
  fprintf(' Iter: %4d mod: %.15f\n', i, s);
  psiL = psi*wP;
  psi = psiL*P';
end

return

legBin = fopen('leg.bin', 'r');
dims = fread(legBin, [1 2], 'int32')
Leg = fread(legBin, [ 2*dims(1), dims(2)], 'double');
fclose(legBin);

Leg = Leg(1:2:end, :);
max(max(Leg - P'))

wLegBin = fopen('weighted_ass_leg.bin', 'r');
dims = fread(legBin, [1 2], 'int32');
wLeg = fread(legBin, [ 2*dims(1), dims(2)], 'double');
fclose(wLegBin);

wLeg = wLeg(1:2:end, :);

max(max(wP - wLeg))

