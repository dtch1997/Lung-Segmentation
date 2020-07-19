% Preprocessing script for LiTS data. Resamples all the data, with
% anti-aliasing.

rootDir = '/media/truecrypt1/blaine/data/LiTS/';
dataDir = fullfile(rootDir, 'train');

% Dependencies
addpath(genpath('~/ctOrganSegmentation'))

% Choose the window size for cropping the final images, in voxels. The
% image is cropped rectangularly to eliminate windows with no labels.
cropWinSiz = [120 120 160];

% Choose whether or not to replace existing results
replace = true;
disp(replace)

% Choose whether to make kidneys convex (warning: this seems bad)
convexKidney = false;
disp(convexKidney)

% Choose wheter ot load the lungs or recompute them
loadLungs = true;
disp(loadLungs)

% Choose whether to load the bones or recompute them (takes a long time)
loadBones = true;
disp(loadBones)

% Target resolution
res = 3; % in mm

% Get a table for the organ labels
maskTagsLabels = organLabels;

% Find the input data folders
manualDir = fullfile(rootDir, 'manual_annotations');
numOrgans = size(maskTagsLabels, 1);
manualMaskDirs = cell(numOrgans, 1);
for i = 1 : length(maskTagsLabels)
   dirName = fullfile(manualDir, maskTagsLabels{i, 1});
   if (exist(dirName, 'dir') ~= 7)
      error(['Could not find directory ' dirName]) 
   end
   manualMaskDirs{i} = dirName;
end

% Destination folders
destDir = fullfile(rootDir, ['labels-' num2str(res) 'mm']); % For storing the final output, interpolated labels
labelsDir = fullfile(rootDir, 'labels'); % For storing the full-res labels
autoDir = fullfile(rootDir, 'automatic_annotations');
bonesAutoDir = fullfile(autoDir, 'bones_auto'); % For storing automatic bones
lungsAutoDir = fullfile(autoDir, 'lungs_auto'); % For storing automatic lungs
convexKidneysDir = fullfile(rootDir, 'convexKidneys'); % For storing post-processed kidneys

% Make the output directories, if they don't exist
if ~exist(destDir, 'dir')
    mkdir(destDir)
end
if ~exist(labelsDir, 'dir')
    mkdir(labelsDir)
end
if ~exist(bonesAutoDir, 'dir')
    mkdir(bonesAutoDir)
end
if ~exist(convexKidneysDir, 'dir')
    mkdir(convexKidneysDir)
end

% Get the image volumes and masks
origDir = pwd;
cd(dataDir)
volFiles = sort(strsplit(ls('volume*.nii.gz')));
cd(origDir)

% Remove the leading empty string
if isempty(volFiles{1})
    volFiles = volFiles(2 : end);
end

% Get the organ processing order. Put lungs and bones last
organLabels_ = cell2mat(maskTagsLabels(:, 2));
organOrder = nan(size(organLabels_'));
organOrder(end - 1) = getOrganLabel('lung');
organOrder(end) = getOrganLabel('bone');
organOrder(isnan(organOrder)) = setdiff(organLabels_, ...
    organOrder(~isnan(organOrder)));

% Process each case
numCases = length(volFiles);
allManualAnnotations = false(numCases, 1);
targetUnits = repmat(res, [3 1]);
% gcp
% parfor i = 1 : length(volFiles)
for i = 1 : numCases
    
    % File tags and output labels for each organ
    % Conflicting labels are resolved in ascending order i.e. label 1
    % overwrites label 2
    
    % Get the corresponding mask names
    volName = volFiles{i};
    volPath = fullfile(dataDir, volName);
    numLabels = size(maskTagsLabels, 1);
    maskNames = cell(numLabels, 1);
    for j = 1 : size(maskTagsLabels, 1)
        maskNames{j} = regexprep(volName, 'volume', maskTagsLabels{j, 1});
    end
    
    % Get the output filenames
    tag = [num2str(res) 'mm-'];
    resultVolPath = fullfile(destDir, [tag volName]);
    resultLabelsProcPath = regexprep(resultVolPath, 'volume', 'labels');
    resultLabelsPath = fullfile(labelsDir, regexprep(volName, 'volume', ...
        'labels'));
    
    % Check if the result was already written
    if exist(resultVolPath, 'file') && ...
            exist(resultLabelsPath, 'file') && ~replace
        warning(['File ' volPath ' previously processed, skipping this '...
            'case...'])
        continue
    end
    
    % Read the volume
    [vol, units] = imRead3D(volPath);
    
    % Process each organ
    valid = true;
    labels = zeros(size(vol));
    manualAnnotations = true(numOrgans, 1);
    for j = organOrder
        organLabel = maskTagsLabels{j, 2};
        maskPath = fullfile(manualMaskDirs{j}, maskNames{j});
        
        % Take the mask, if it exists. (Use this to insert manual lungs &
        % bones for the validation set.)
        if exist(maskPath, 'file')
            [mask, maskUnits] = imRead3D(maskPath);
            
            % Check dimensions and units
            if ~isequal(size(vol), size(mask))
                warning(['Dimensions do not match for file ' maskPath])
                valid = false;
                break
            end
            if norm(units - maskUnits) > 1E-3
                warning(['Units do not match for file ' maskPath ...
                    ' proceeding anyways...'])
            end
            
            % Print progress for the organs which are normally automatic
            switch organLabel
                case getOrganLabel('lung')
                    disp(['Loaded manual lung ' maskPath])
                case getOrganLabel('bone')
                    disp(['Loaded manual bones ' maskPath])
            end
        else
            % Process each mask depending on organ type
            switch organLabel
                case getOrganLabel('lung')
                    % Find the lungs using morphology, unless they were
                    % already computed
                    manualAnnotations(j) = false;
                    lungsLoaded = false;
                    if loadLungs
                        % Get the output path
                        resultLungsPath = fullfile(lungsAutoDir, ...
                                regexprep(volName, 'volume', 'lungs-auto'));
                        
                        % Load the existing lungs, don't re-compute
                        try
                            mask = imRead3D(resultLungsPath);
                            lungsLoaded = any(mask(:) > 0);
                            disp(['Loaded lungs from ' resultLungsPath])
                        catch ME
                            warning(['Failed to load lungs from ' ...
                                resultLungsPath ', recomputing...'])
                            lungsLoaded = false;
                        end
                    end
                    
                    if ~lungsLoaded
                        disp(['Automatically extracting lung ' maskPath])
                        mask = findLungsCT(vol, units, labels > 0);
                        imWrite3D(resultLungsPath, mask, units);
                        disp(['Saved automatic lungs to ' resultLungsPath])
                    end
                case getOrganLabel('bone')
                    
                    manualAnnotations(j) = false;
                    
                    % Get the output path
                    resultBonesPath = fullfile(bonesAutoDir, ...
                        regexprep(volName, 'volume', 'bones-auto'));
                    
                    bonesLoaded = false;
                    if loadBones
                        % Load the existing bones, don't re-compute
                        try
                            mask = imRead3D(resultBonesPath);
                            bonesLoaded = any(mask(:) > 0);
                            disp(['Loaded bones from ' resultBonesPath])
                        catch ME
                            warning(['Failed to load bones from ' ...
                                resultBonesPath ', recomputing...'])
                            bonesLoaded = false;
                        end
                    end
                    
                    if ~bonesLoaded
                        disp(['Computing bones for ' resultBonesPath])
                        
                        % Find the bones using morphology
                        mask = findBonesCT(vol, units, labels > 0);
                        
                        % Save the original bones, since these take a while to
                        % compute
                        imWrite3D(resultBonesPath, mask, units);
                        disp(['Saved automatic bones to ' resultBonesPath])
                    end
                case getOrganLabel('brain')
                    warning(['Missing brain mask ' maskPath ...
                        '. Assuming it''s empty...'])
                    mask = false(size(vol));
                      case getOrganLabel('gallbladder')
                    warning(['Missing gallbladder mask ' maskPath ...
                        '. Assuming it''s empty...'])
                    mask = false(size(vol));
                otherwise
                    valid = false;
                    error(['Missing mask ' maskPath])
            end
        end
        
        % Special post-processing
        if organLabel == getOrganLabel('kidneys') && ...
                convexKidney
            % Take the convex hulls of the 2 largest CC's
            stats = regionprops(mask > 0, 'Area', 'Image', ...
                'BoundingBox');
            [~, regionSort] = sort(extractfield(stats, 'Area'), ...
                'descend');
            mask = false(size(mask));
            for k = 1 : min(length(regionSort), 2)
                stat = stats(regionSort(k));
                mask = fillIdx(mask, convHull3D(stat.Image),...
                    round(stat.BoundingBox([2 1 3])));
            end
        end
        
        % Write the mask into the total organ labels, do not over-write
        % previous labels
        labels(mask > 0 & labels == 0) = organLabel;
        clear mask maskUnits
        
    end
    
    % Check if any of the segmentations failed
    if ~valid
        warning('Skipping this case...')
        continue
    end
    
    % Mark whether the annotations are all manual or not
    allManualAnnotations(i) = all(manualAnnotations);
    
    % Save the labels
    writeNiiLike(resultLabelsPath, labels, units, volPath);
    
    % Interpolate the results
    volProc = resampleAA(vol, units, res);
    labelsProc = volResize(labels, size(volProc), 'nearest', 0);
    
    % Crop the scan to eliminate empty windows, in the final resolution
    origSiz = size(labelsProc);
    [I, J, K] = ind2sub(origSiz, find(labelsProc > 0));
    minFind = [min(I) min(J) min(K)];
    maxFind = [max(I) max(J) max(K)];
    cropMin = max(minFind - cropWinSiz + 1, 1);
    cropMax = min(maxFind + cropWinSiz - 1, size(labelsProc));
    volProc = volProc(cropMin(1) : cropMax(1), cropMin(2) : cropMax(2), ...
        cropMin(3) : cropMax(3));
    labelsProc = labelsProc(cropMin(1) : cropMax(1), ...
        cropMin(2) : cropMax(2), cropMin(3) : cropMax(3));
    disp(['Volume ' num2str(i)])
    disp(['Original size (interpolated): ' num2str(origSiz)])
    disp(['Cropped size: ' num2str(size(labelsProc))])
    
    % Save the interpolated results
    imWrite3D(resultVolPath, volProc, targetUnits)
    imWrite3D(resultLabelsProcPath, labelsProc, targetUnits)    
end

% Write the marker 'manualAnnotations' file
save(fullfile(labelsDir, 'manualAnnotations.mat'), 'allManualAnnotations')
disp('Done!')
