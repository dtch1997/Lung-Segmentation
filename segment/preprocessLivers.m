% Preprocessing script for LiTS data. This extracts the livers, and sets
% all other pixels to -inf. Resamples all the data, with anti-aliasing.

% Dependencies
addpath(genpath('~/aimutil'))

% Data folder
rootDir = '/media/truecrypt1/blaine/data/LiTS';  % Change this for LiTS data
dataDirName = 'train';
%rootDir = '/media/truecrypt1/blaine/data/mint/processed'; % For mint data
%dataDirName = 'baselines_with_liver_and_non_target';
dataDir = fullfile(rootDir, dataDirName);

% Target resolution
res = 1; % in mm

% Replace existing masks
replace = true;

% Make voronoi diagrams
makeVoronoi = false;

% Make fuzzy membership 
makeFuzzy = false;

% Destination folder
destDir = fullfile(rootDir, ['lesions-' num2str(res) 'mm']); % Change this for LiTS data
%destDir = fullfile(rootDir, [dataDirName '-' num2str(res) 'mm']); % For mint data
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
%gcp
%parfor i = 1 : length(volFiles)
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
        
    % Crop and preprocess the image
    [volProc, maskProc] = preprocessMaskVolume(vol, units, res, mask);
    
    % Set lesions to the label 1, and background to -1
    maskProc = maskProc - 1;
    
    % Optionally the voronoi diagram of the objects
    lesionsProc = maskProc == 1;
    if makeVoronoi
        voronoiLabels = objVoronoi(lesionsProc);
    end
    if makeFuzzy
        [fuzzyWeights, fuzzyLabels] = fuzzyMembership(lesionsProc);
    end
    
    % Save the result
    imWrite3D(resultVolPath, volProc, targetUnits)
    imWrite3D(resultMaskPath, maskProc, targetUnits)
    if makeVoronoi
        imWrite3D(resultVoronoiPath, voronoiLabels, targetUnits)
    end
    if makeFuzzy
        % Write the fuzzy weights
        imWrite3D(resultFuzzyWeightsPath, fuzzyWeights, targetUnits)
        imWrite3D(resultFuzzyLabelsPath, fuzzyLabels, targetUnits);
    end
end
