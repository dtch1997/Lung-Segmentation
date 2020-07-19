function [volProc, maskProc, cropIdx, volProcSiz] = preprocessMaskVolume(...
    vol, units, res, mask, maskUnits)
% Crop and preprocess a volume using a mask. Resamples to the desired
% resolution, and sets everything outside the mask to -inf. If a mask is
% not provided, this just resamples the input. If the resolution is not
% provided, this simply crops instead of resampling.
%
% Returns the bounding box for cropping, if any was used. Also returns the
% total size of the un-cropped volume in the new resolution.

if nargin < 4
    mask = [];
end
if nargin < 5 || isempty(maskUnits)
    maskUnits = units;
end

% If provided a mask, crop the volume using the mask bounding box
if ~isempty(mask)
    % Crop the mask
    %stats = regionprops(mask > 0, 'BoundingBox');
    %bbMask = ceil(stats.BoundingBox);
    [bbMin, bbMax] = boundingBox(mask > 0);
    bbMask = [bbMin bbMax - bbMin];
    bbMask = bbMask([2 1 3], :); % Swap to match with regionprops() format
    maskCrop = cropBB(mask, bbMask);
    
    % Transfer the mask bounding box to the volume, round it
    bbVolFloat = bsxfun(@times, reshape(bbMask, [3 2])' - 1, ...
        (maskUnits ./ units)')' + 1;
    bbStart = floor(bbVolFloat(:, 1));
    bbExtent = ceil(bbVolFloat(:, 2));
    bbExtent = min(bbExtent, size(vol)' - bbStart);
    bbVol = [bbStart bbExtent];
    
    % Crop the volume
    volCrop = cropBB(vol, bbVol);
else
    volCrop = vol;
    maskCrop = mask;
    bbVol = [1 1 1 size(vol, 2) size(vol, 1) size(vol, 3)]; 
end

% Interpolate the volume, if res was given
if ~isempty(res)
    growFactors = units' / res;
    volProc = resampleAA(volCrop, units, res);
else
    volProc = volCrop;
end

% Interpolate the mask, if both it and res were given
if isempty(mask)
    maskProc = [];
elseif isempty(res)
    maskProc = maskCrop;
else
    maskProc = volResize(maskCrop, size(volProc), 'nearest', 0);
end

% Return the cropping indices, in the new resolution
volSiz = [size(vol, 1) size(vol, 2) size(vol, 3)];
volProcSiz = ceil(volSiz .* growFactors);
cropIdx = round(([bbVol(2) bbVol(1) bbVol(3)] - 1) .* growFactors + 1);

end