function file_inference(inName, outName, modelPbPath, modelPklPath, res)
% runs inference.py on files. Helper for inference.m.

% Inference script location, relative to this script
scriptDir = fileparts(mfilename('fullpath'));
inferencePath = fullfile(scriptDir, '..', 'utils', 'CTH_seg_inference', 'inference.py');

% Run the model
ret = runCmd(['python ' inferencePath ' ' modelPbPath ' ' modelPklPath ...
    ' ' inName ' ' outName ' ' num2str(res)]);
if ret ~= 0
    error(['Python returned code ' num2str(ret)])
end

end


function status = runCmd(cmd)
%runCmd helper function to run a command in the default system environment,
% without Matlab's changes to the environment variables. Also makes sure to
% run this as 'user'

% Strip the LD_LIBRARY_PATH environment variable of Matlab directories
ldPathVar = 'LD_LIBRARY_PATH';
oldLdPath = getenv(ldPathVar);
newLdPath = regexprep(oldLdPath, '[^:]*MATLAB[^:]*:*', '', 'ignorecase');
setenv(ldPathVar, newLdPath);

% Run the command
status = system(cmd);

% Return the environment to its previous state
setenv(ldPathVar, oldLdPath);

end
