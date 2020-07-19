function [predPET, predCT] = findOrgansPetCt(ct, ctUnits, pet)
% Given a co-registered PET/CT pair, detect organs on the CT and then
% expand those labels ot Pet.

% Dependencies
addpath(genpath('~/aimutil'))
addpath(genpath('~/CTH_seg/matlab'))

% Organ segmenter parameters
nnPbPath = '/home/blaine/CTH_seg/frozen/CT_Organ.pb';
nnPklPath = '/home/blaine/CTH_seg/frozen/CT_Organ.params.pkl';
nnRes = 5;

% Get the class predictions
predCT = inference(ct, ctUnits, nnPbPath, nnPklPath, nnRes);
 
% Resize the class predictions
predPET = volResize(predCT, size(pet), 'nearest', -1);
 
end
