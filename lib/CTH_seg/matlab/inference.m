function [prob, volProc, labels] = inference(vol, units, modelPbPath, ...
    modelPklPath, res, mask, maskUnits, hashDir)
% Script to run the model. Clips the result using the ROI mask, if
% provided. If hashDir is provided, stores the output in that directory and
% looks it up for future use.

if nargin < 6
    mask = [];
end
if nargin < 7 || isempty(maskUnits)
    maskUnits = units;
end
haveHashDir = nargin >= 8 && ~isempty(hashDir);

% Resample the volume, with anti-aliasing
try %XXX catch errors for debugging
[volProc, maskProc, cropIdx, outSiz] = preprocessMaskVolume(vol, units, ...
    res, mask, maskUnits);
catch
    keyboard
end
outUnits = repmat(res, size(units));

% Initialize the hash table
if haveHashDir
   hasher = nnHasher(hashDir);
   hashStr = hasher.getInString(volProc, res, modelPbPath);
end

% Query the hash table
if haveHashDir && hasher.hasData(hashStr)
    
    %TODO: To improve this, we could hash the actual content of the .pb
    % file, check its modification time, and redo the the hash only if
    % the file was modified. Could keep track of hashed .pb files in
    % the nnHasher class.
    labels = hasher.get(hashStr);
else
    if haveHashDir
        warning('NN hash missed. Recomputing...')
    end
    
    % Write the input
    inName = [tempname '.nii'];
    imWrite3D(inName, volProc, outUnits);
    
    % Run the model
    outName = [tempname '.nii'];
    file_inference(inName, outName, modelPbPath, modelPklPath, res)
    
    % Read the result
    labels = imRead3D(outName);
    
    % Delete the temporary files
    delete(inName)
    delete(outName)
    
    % Put the output in the hash table
    if haveHashDir
        hasher.put(hashStr, labels);
    end
    
end

% Sanity check
assert(isequal(size(volProc), size(labels)))

% If a mask was provided, set everything outside mask to -1
if ~isempty(maskProc)
    labels(~maskProc) = -1;
end

% Pad the output volume to account for cropping
if isequal(outSiz, size(labels))
    assert(all(cropIdx == 1))
    prob = labels;
else
    prob = fillIdx(-ones(outSiz), labels, cropIdx);
end

end