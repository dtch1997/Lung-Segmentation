% Preprocessing script for LiTS data. This extracts the livers, and sets
% all other pixels to -inf. Resamples all the data, with anti-aliasing.

% Dependencies
addpath(genpath('~/bbrister/lung_segmentation/lib/aimutil'))
addpath(genpath('~/bbrister/lung_segmentation/lib/cth_seg'))
  
% Data folder
rootDir = '/data/lidc';
dataDirName = 'LIDC-NIFTI/debug';
dataDir = fullfile(rootDir, dataDirName);

% Target resolution
res = 1; % in mm

% Replace existing masks
replace = true;

% Destination folder
destDir = fullfile(rootDir, ['lesions-' num2str(res) 'mm']); % Change this for LiTS data
if ~exist(destDir, 'dir')
    mkdir(destDir)
end

% Get the image volumes and masks
wd = pwd;
cd(dataDir)
volFiles = sort(strsplit(ls('*volume*.nii.gz')));
cd(wd)

% Remove the leading empty string
if isempty(volFiles{1})
    volFiles = volFiles(2 : end);
end

% Process each pair
targetUnits = repmat(res, [3 1]);
for i = 1 : length(volFiles)
    
    % Get the corresponding mask name and ensure it exists
    volName = volFiles{i};
    maskName = regexprep(volName, 'volume', 'segmentation');
    maskPath = fullfile(dataDir, maskName);
    volPath = fullfile(dataDir, volName);
    if ~exist(maskPath, 'file')
        warning(['Cannot find file ' maskPath ' skipping this case...'])
        continue
    end
    
    % Check if the result was already written
    tag = [num2str(res) 'mm-'];
    resultVolPath = fullfile(destDir, [tag volName]);
    resultMaskPath = fullfile(destDir, [tag maskName]);
    resultVoronoiPath = regexprep(resultMaskPath, 'segmentation', ...
        'voronoi');
    resultFuzzyWeightsPath = regexprep(resultMaskPath, 'segmentation', ...
        'fuzzyWeights');
    resultFuzzyLabelsPath = regexprep(resultMaskPath, 'segmentation', ...
        'fuzzyLabels');
    if ~replace && exist(resultVolPath, 'file') && ...
            exist(resultMaskPath, 'file') && ...
        (~makeVoronoi || exist(resultVoronoiPath, 'file')) && ...
        (~makeFuzzy || (exist(resultFuzzyWeightsPath, 'file') && ...
            exist(resultFuzzyLabelsPath, 'file')));
        warning(['File ' volPath ' previously processed, skipping this '...
            'case...'])
        continue
    end
    
    % Read the images
    [vol, units] = imRead3D(volPath);
    [mask, maskUnits] = imRead3D(maskPath);
    
    % Verify the mask
    if ~any(mask(:))
        warning(['Mask is empty for file ' maskPath ' skipping this '...
            'case...'])
        continue
    end
    
    % Check dimensions and units
    if ~isequal(size(vol), size(mask))
        warning(['Dimensions do not match for file ' maskPath ...
            ' skipping this case...'])
        continue
    end
    if norm(units - maskUnits) > 1E-3
        warning(['Units do not match for file ' maskPath ' proceeding '...
            'anyways...'])
    end

    % Get organ segmentor predictions
     
    % Crop and preprocess the image
    
    % Set lesions to the label 1, and background to -1
    
    % Save the result
end
