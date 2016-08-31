
function [] = FH2Matlab()

global FH2Data

fprintf(' From FH2Matlab\n')

%{
if mod(FH2Data.time.steps, 10) == 0
  PlotCRP();
  if FH2Data.CRP.calculate_CRP == 1
    CRP = FH2Data.CRP;
    save(FH2Data.options.CRPMatFile, 'CRP');
  end
end
%}

if mod(FH2Data.time.steps, 100) == 0
  PlotPotWave(FH2Data.r1, FH2Data.r2, FH2Data.pot, ...
	      FH2Data.OmegaStates.wave_packets(:,:,:,2));
end

return
