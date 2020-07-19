% Script to generate results for the LiTS challenge

% Settings
replace = false;

% Dependencies
addpath(genpath('~/ctOrganSegmentation'))
addpath(genpath('dicomrt2matlab'))
addpath(genpath('~/CTH_seg/matlab'))
addpath(genpath('~/aimutil'))

% Organ segmenter parameters
nnOrganDir = '/home/blaine/CTH_seg/frozen/organ_seg_multi_with_aug';
nnOrganPbPath = fullfile(nnOrganDir, 'CT_Organ_3mm_extended_Unet_multi.pb');
nnOrganPklPath = strrep(nnOrganPbPath, '.pb', '.params.pkl');
nnOrganRes = 3;

% Lesion segmenter parameters
nnLesionDir = '/home/blaine/CTH_seg/frozen/liver_lesion_unet';
nnLesionPbPath = fullfile(nnLesionDir, 'CT_Liver_Lesion_Unet.pb');
nnLesionPklPath = strrep(nnLesionPbPath, '.pb', '.params.pkl');
nnLesionRes = 1;

% Data locations
rootDir = '/media/truecrypt1/blaine/data/LiTS/';
dataDir = fullfile(rootDir, 'test');
outDir = fullfile(rootDir, 'experiments', 'test');

if ~exist(outDir, 'dir')
    mkdir(outDir)
end

% Number of classes for each experiment
numNnClasses = size(organLabels, 1) + 1;

% Get the test files
testFiles = dir(dataDir);
testFilenames = {testFiles(:).name};

% Process each file
for filename = testFilenames
    filename = filename{1};
    
    % Skip non-Nifti files
    [~, ~, ext] = fileparts(filename);
    if ~strcmp(ext, '.nii')
        warning(['Skipping file ' filename])
        continue
    end
    
    % Get the output path and check if it exists
    labelsName = strrep(filename, 'volume', 'segmentation');
    labelsPath = fullfile(outDir, labelsName);
    if ~replace && exist(labelsPath, 'file')
        disp(['Found file ' labelsPath])
        continue
    end
    
    % Run deep learning on the organs
    volPath = fullfile(dataDir, filename);
    file_inference(volPath, labelsPath, nnOrganPbPath, ...
        nnOrganPklPath, nnOrganRes);
    
    % Read the results
    [ct, ctUnits] = imRead3D(volPath);
    organLabels = imRead3D(labelsPath);
    
    % Get the liver label
    mask = organLabels == getOrganLabel('liver');
    clear organLabels
    
    % Run inference on the lesions
    [lesionLabels, ctProc, ctProcLabels] = inference(ct, ctUnits, ...
        nnLesionPbPath, nnLesionPklPath, nnLesionRes, mask);
    
    % Resize the outputs
    lesionLabels = volResize(lesionLabels, size(ct), 'nearest', 0);
    
    % Write in the lesion labels
    finalLabels = single(mask);
    finalLabels((lesionLabels == 1) & mask) = 2; % 1 = liver, 2 = lesion
    clear lesionLabels
    
    % Write the results
    writeNiiLike(labelsPath, finalLabels, ctUnits, volPath);
    disp(['Wrote ' labelsPath])
    clear ct mask finalLabels
end
