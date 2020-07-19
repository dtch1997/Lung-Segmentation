function processPetCtHelper(opts)
% Does final processing for processPetCt.m. This part was separated into
% its own function to allow for recursion.

% Get the organ bounding box
[mX, mY, mZ] = ind2sub(size(opts.organ), find(opts.organ));
xMin = min(mX);
yMin = min(mY);
zMin = min(mZ);
xMax = max(mX);
yMax = max(mY);
zMax = max(mZ);

% Get the resizing and registration matrices
matPet2Res = [diag(opts.res ./ opts.petUnits) zeros(3, 1)];
matCt2Res = composeAffine(opts.matPet2Ct, matPet2Res);

% Register the corners of the cube
[cornerX, cornerY, cornerZ] = ndgrid([xMin xMax], [yMin yMax], ...
    [zMin zMax]);
ctCornerPts = [cornerX(:) cornerY(:) cornerZ(:)];
resCornerPts = applyAffine(invertAffine(matCt2Res), ctCornerPts);

% Get the bounding box in output space
bbMin = floor(min(resCornerPts, [], 1));
bbMax = ceil(max(resCornerPts, [], 1));

% Get the cropping matrix
matCrop = [eye(3) (bbMin - 1)'];

% Form the total registration matrices
matPet2ResCrop = composeAffine(matPet2Res, matCrop);
matCt2ResCrop = composeAffine(matCt2Res, matCrop);

% Get the output size
outSiz = bbMax - bbMin + 1;

% Register the organ labels
organ = invAffine3D(matCt2ResCrop, opts.organ, outSiz, 'nearest');

% Split the lungs, if in lung mode
if opts.lungMode
    % Cut the lungs at the median
    organInds = find(organ);
    [X, ~, ~] = ind2sub(size(organ), organInds);
    xCut = median(X);
    mask = organ;
    mask(organInds(X == xCut)) = false;
    clear X
    
    % Take the two largest objects
    stats = regionprops(mask, 'Area', 'PixelIdxList');
    if length(stats) < 2 %XXX
        keyboard
    end
    [~, sortIdx] = sort([stats.Area], 'descend');
    keepIdx = sortIdx(1 : 2);
    seedIdxList = {stats(keepIdx).PixelIdxList};
    
    % Get the center (mean) of each object
    centers = zeros(2, 3);
    for j = 1 : 2
        [X, Y, Z] = ind2sub(size(organ), seedIdxList{j});
        centers(j, 1) = mean(X);
        centers(j, 2) = mean(Y);
        centers(j, 3) = mean(Z);
    end
    
    % Convert the centers to indices
    centers = round(centers);
    centerInds = sub2ind(size(organ), centers(:, 1), centers(:, 2), ...
        centers(:, 3));
    
    % Take the geodesic distance to each center
    numCenters = length(centerInds);
    dists = zeros([size(organ) numCenters]);
    for i = 1 : numCenters
        center = centerInds(i);
        if ~organ(center)
           % If the center of mass is not in the organ, take the projection
           [X, Y, Z] = ndgrid(1 : outSiz(1), 1 : outSiz(2), 1 : outSiz(3));
           [xC, yC, zC] = ind2sub(size(organ), center);
           centerDists = zeros([size(organ) 3]);
           centerDists(:, :, :, 1) = abs(X - xC);
           centerDists(:, :, :, 2) = abs(Y - yC);
           centerDists(:, :, :, 3) = abs(Z - zC);
           centerDist = max(centerDists, [], 4);
           centerDist(~organ) = inf;
           [~, projectIdx] = min(centerDist(:));
           center = projectIdx(1);
        end
        dists(:, :, :, i) = bwdistgeodesic(organ, center);
    end
    masks = false(size(dists));
    masks(:, :, :, 1) = (dists(:, :, :, 1) < dists(:, :, :, 2)) & organ;
    masks(:, :, :, 2) = ~masks(:, :, :, 1) & organ;
    assert(size(masks, 4) == 2) % or else the previous line is wrong
    
    % Finish processing on each mask
    for i = 1 : size(masks, 4)
        % Make a copy of the input
        newOpts = opts;
        
        % Set up the output names
        fieldNames = {'outPetFilename', 'outCtFilename', ...
            'outLabelsFilename'};
        if ~isempty(opts.outVoronoiFilename)
            fieldNames{end + 1} = 'outVoronoiFilename';
        end
        if ~isempty(opts.outFuzzyFilename)
            fieldNames{end + 1} = 'outFuzzyFilename';
        end
        oldSubStr = 'lung';
        newSubStr = [oldSubStr num2str(i)];
        for j = 1 : length(fieldNames)
            fieldName = fieldNames{j};
            [pathName, baseName, ext] = fileparts(newOpts.(fieldName));
            subBaseName = strrep(baseName, oldSubStr, newSubStr);
            newOpts.(fieldName) = fullfile(pathName, [subBaseName ext]);
        end
        
        % Register the volumes scans to the coordinates of the organ mask
        newOpts.pet = warpAA(matPet2ResCrop, opts.pet, outSiz);
        newOpts.ct = warpAA(matCt2ResCrop, opts.ct, outSiz);
        newOpts.lesions = invAffine3D(matPet2ResCrop, opts.lesions, ...
            outSiz, 'nearest');
        newOpts.matPet2Ct = [eye(3) zeros(3, 1)];
        newOpts.petUnits = repmat(opts.res, size(opts.petUnits));
        newOpts.ctUnits = newOpts.petUnits;
        
        % Set up other options
        newOpts.lungMode = false;
        newOpts.organ = masks(:, :, :, i);
        
        % Go
        processPetCtHelper(newOpts);
    end
    return % All lungs finished
end

% Register the PET and save
outUnits = repmat(opts.res, [3 1]);
if opts.replacePet || ~exist(opts.outPetFilename, 'file')
    pet = warpAA(matPet2ResCrop, opts.pet, outSiz);
    imWrite3D(opts.outPetFilename, pet, outUnits);
    disp(['Wrote ' opts.outPetFilename])
end

% Register the CT and save
if opts.replaceCt || ~exist(opts.outCtFilename, 'file')
    ct = warpAA(matCt2ResCrop, opts.ct, outSiz);
    imWrite3D(opts.outCtFilename, ct, outUnits);
    disp(['Wrote ' opts.outCtFilename])
end

% Exist early if only lesions remain
if ~opts.replaceLabels && exist(opts.outLabelsFilename, 'file')
    return
end

% Load the lesions, register
lesions = invAffine3D(matPet2ResCrop, opts.lesions, ...
    outSiz, 'nearest');

% Create the mask
if strcmp(opts.organMasked, 'exact')
    mask = organ;
elseif strcmp(opts.organMasked, 'closed')
    closeWidth = ceil(opts.closeMm / opts.res);
    if mod(closeWidth, 2) == 0
        closeWidth = closeWidth + 1;
    end
    closeSiz = repmat(closeWidth, [3 1]);
    center = 1 + (closeWidth - 1) / 2;
    ball = ballMask(closeSiz, center, (closeWidth - 1) / 2);
    mask = bwCloseN(organ, ball);
elseif strcmp(opts.organMasked, 'convex')
    addpath(genpath('~/ctOrganSegmentation'))
    mask = convHull3D(organ) & (organ | ~organs); % Convex hull
    % relative to the given organ and the background
elseif strcmp(opts.organMasked, 'none')
    mask = false(size(lesions));
else
    error(['Unrecognized setting for organMasked: ' organMasked])
end

% Add lesions touching the desired organ
mask = imreconstruct(mask, mask | (lesions > 0));

% Set everything outside the organ mask to -1
lesions(~mask) = -1;

% Save the labels
imWrite3D(opts.outLabelsFilename, lesions, outUnits);
disp(['Wrote ' opts.outLabelsFilename])

% Optionally compute the voronoi and save
if ~isempty(opts.outVoronoiFilename)
   voronoiLabels = objVoronoi(lesions == 1);
   imWrite3D(opts.outVoronoiFilename, voronoiLabels, outUnits);
   disp(['Wrote ' opts.outVoronoiFilename])
end

% Optionally compute the fuzzy labels and save
if ~isempty(opts.outFuzzyFilename)
   fuzzyLabels = fuzzyMembership(lesions == 1);
   imWrite3D(opts.outFuzzyFilename, fuzzyLabels, outUnits);
   disp(['Wrote ' opts.outFuzzyFilename])
end

end