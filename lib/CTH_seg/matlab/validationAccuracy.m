% Script to generate segmentation accuracy results. Runs the automatic
% labeling algorithms as well as the deep learning model

% Dependencies
addpath(genpath('~/ctOrganSegmentation'))
addpath(genpath('dicomrt2matlab'))
addpath(genpath('~/aimutil'))

% Organ segmenter parameters
nnDir = '/home/blaine/CTH_seg/frozen/no_augment_cross_ent_iter_17600';
nnOrganPbPath = fullfile(nnDir, 'CT_Organ_3mm_no_augment.pb');
nnOrganPklPath = fullfile(nnDir, 'CT_Organ_3mm_no_augment.params.pkl');
nnOrganRes = 3;

% Load the NN results?
loadNN = false

% Validation set indices
valIdx = [0 : 16, 20 : 22, 134]

% Automatically-generated labels
autoLabelNames = {'lung', 'bone'};
autoLabelFcns = {@findLungsCT, @findBonesCT};

% Data locations
rootDir = '/media/truecrypt1/blaine/data/LiTS/';
dataDir = fullfile(rootDir, 'train');
labelsDir = fullfile(rootDir, 'labels');
outDir = fullfile(rootDir, 'experiments', 'cross_ent_no_aug');

if ~exist(outDir, 'dir')
    mkdir(outDir)
end

% Number of classes for each experiment
numAutoClasses = length(autoLabelNames) + 1;
numNnClasses = size(organLabels, 1) + 1;

% Process each image in the validation set, record Dice scores
numVal = length(valIdx);
autoDices = zeros(numVal, numAutoClasses);
nnDices = zeros(numVal, numNnClasses);
for i = 1 : numVal
    %TODO load the volume, run automatic labeling (and save for later
    %use), run deep learning labeling, compute Dice, all at native
    %resolution
    % Load the volume
    volIdx = valIdx(i);
    volName = sprintf('volume-%d.nii.gz', volIdx);
    volInName = fullfile(dataDir, volName);
    [im, units] = imRead3D(volInName);
    
    % Write the volume, for rendering later
    volOutName = fullfile(outDir, volName);
    imWrite3D(volOutName, im, units);
    
    % Load the 'ground truth' labels
    labelsGtName = fullfile(labelsDir, sprintf('labels-%d.nii.gz', ...
        volIdx));
    labelsGt = imRead3D(labelsGtName);
    
    % Get the 'ignore' labels for automatic labeling
    ignore = labelsGt > 0 & labelsGt ~= getOrganLabel('lung') & ...
        labelsGt ~= getOrganLabel('bone');
    
    % Run automatic labeling on each organ, if we don't have it already
    autoLabels = zeros(size(labelsGt));
    autoGt = zeros(size(labelsGt));
    for j = 1 : length(autoLabelNames)
        autoLabelName = autoLabelNames{j};
        autoLabelPath = fullfile(outDir, sprintf('auto-%s-%d.nii.gz', ...
            autoLabelName, volIdx));
        if exist(autoLabelPath, 'file')
            disp(['Reading ' autoLabelPath])
            autoLabel = imRead3D(autoLabelPath) > 0;
        else
            % Compute the automatic labels
            disp(['Computing ' autoLabelPath '...'])
            autoLabel = autoLabelFcns{j}(im, units, ignore);
            
            % Save them for later
            imWrite3D(autoLabelPath, autoLabel, units);
            disp(['Wrote ' autoLabelPath])
        end
        
        % Add the label to 'ignore' for the other ones
        ignore = ignore | autoLabel;
        
        % Populate the autoLabels, autoGt arrays
        autoLabels(autoLabel) = j;
        autoGt(labelsGt == getOrganLabel(autoLabelName)) = j;
    end
    
    % Compute Dice score for auto labels, free memory
    autoDices(i, :) = dice(autoLabels, autoGt, numAutoClasses);
    clear ignore autoLabel autoLabels autoGt
    
%     nnLabelsPath = fullfile(outDir, sprintf('nn-%0.1f-%d.nii.gz', ...
%         volIdx, nnOrganRes));
%     if loadNN && exist(nnLabelsPath, 'file')
%         nnLabels = imRead3D(nnLabelsPath);
%         disp(['Read ' nnLabelsPath])
%     else
%         
%         % Run deep learning labeling
%         nnLabelsIso = inference(im, units, nnOrganPbPath, nnOrganPklPath, ...
%             nnOrganRes);
%         matIm2Iso = [diag(units / nnOrganRes) zeros(3, 1)];
%         nnLabels = invAffine3D(matIm2Iso, nnLabelsIso, size(im), 'nearest');
%         
%         % Save it for later
%         imWrite3D(nnLabelsPath, nnLabels, units);
%         disp(['Wrote ' nnLabelsPath])
%     end
%     
%     % Compute Dice score for NN, free memory
%     nnDices(i, :) = dice(nnLabels, labelsGt, numNnClasses);
%     clear im ignore nnLabels nnLabelsIso labelsGt
end

% Print the results
disp('Auto:')
disp(mean(autoDices, 1))
disp('NN:')
disp(mean(nnDices, 1))
