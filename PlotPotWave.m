
% $Id$

function [] = PlotPotWave(r1, r2, pot, psi)

persistent has_PotWavePlot
persistent hpsi

[ psiReal, psiImag ] = WavePacket(psi);

k = 110;

psiReal= psiReal(:,:,k)';

if isempty(has_PotWavePlot)
  
  has_PotWavePlot = 1;
  
  figure(1);
  
  [ ~, hPES ] = contour(r1.r, r2.r, pot(:,:,k)', [ -0.2:0.01:0.3 ]);
  set(hPES, 'LineWidth', 0.75);
  set(hPES, 'LineColor', 'black');
  %axis square
  hold on;
  
  [ ~, hpsi ] = contour(r1.r, r2.r, psiReal, ...
			[ -2.0:0.02:-0.01 0.01:0.02:1.0 ], 'zDataSource', 'psiReal');
  set(hpsi, 'LineWidth', 1.5);
  set(gca, 'CLim', [-0.5, 0.5]);
  %axis square
  colormap jet
  colorbar('vert')
  hold off;
end

refreshdata(hpsi, 'caller');
drawnow

