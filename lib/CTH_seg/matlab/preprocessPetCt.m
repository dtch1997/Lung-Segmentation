% Preprocessing script for LiTS data. This extracts the livers, and sets
% all other pixels to -inf. Resamples all the data, with anti-aliasing.
% Dependencies
addpath(genpath('~/aimutil'))

% Settings
organName = 'lung' % 'lung', 'bone', 'liver'
replacePet = false
replaceCt = false
replaceLabels = true
replaceAny = replacePet || replaceCt || replaceLabels
organMasked = 'closed' % 'exact' takes the exact organ mask. 'convex' takes
% the convex hull, relative to other none organs. 'closed' applies
% morphological closing. 'none' doesn't apply any label mask
closeMm = 25;
makeVoronoi = false; % Make voronoi diagrams?
makeFuzzy = true; % Make fuzzy membership?

% Data folder
rootDir = '/media/truecrypt1/blaine/data/tomomi_PET_CT/';
dataDirName = fullfile('output', 'output_RT2_v4');
dataDir = fullfile(rootDir, dataDirName);

% Output folder
outDir = fullfile(rootDir, 'nn_input_half');

% Target resolution
res = 2; % in mm

% Destination folder
% destDir = fullfile(rootDir, ['lesions-' num2str(res) 'mm']); % Change this for LiTS data
destDir = fullfile(rootDir, [dataDirName '-' num2str(res) 'mm']);
if ~exist(destDir, 'dir')
    mkdir(destDir)
end

% Choose whether to split the lungs apart
breakLungs = true
lungMode = strcmp(organName, 'lung') && breakLungs;

% Look for the PET-CT data
peaksFiles = strsplit(ls(fullfile(dataDir, ['peaks*' organName '.mat'])), ...
    {' ', '\n', '\t'});
peaksFiles = peaksFiles(~cellfun(@isempty, peaksFiles)); % Remove whitespace, which becomes an empty name

% Process each study
%parpool(2)
%gcp
for i = 1 : length(peaksFiles)
    % Get input file names
    peaksFilename = peaksFiles{i};
    [~, peaksBasename] = fileparts(peaksFilename);
    ids = sscanf(peaksBasename, ['peaks_%d_%d_' organName '.mat']);
    patientId = ids(1);
    studyId = ids(2);
    idTag = [num2str(patientId) '_' num2str(studyId)];
    petFilename = fullfile(dataDir, ['pet_' idTag '.nii.gz']);
    ctFilename = strrep(petFilename, 'pet_', 'ct_');
    organsFilename = fullfile(dataDir, ['organs_ct_' idTag '.nii.gz']);
    labelsFilename = strrep(petFilename, 'pet_', 'lesions_pet_');
    
    % Get output file names
    outTag = [idTag '_' organName];
    outPetFilename = fullfile(outDir, ['pet_' outTag '.nii.gz']);
    outCtFilename = fullfile(outDir, ['ct_' outTag '.nii.gz']);
    outLabelsFilename = fullfile(outDir, ['labels_' outTag '.nii.gz']);
    if makeVoronoi
        outVoronoiFilename = fullfile(outDir, ...
            ['voronoi_' outTag '.nii.gz']);
    else
        outVoronoiFilename = [];
    end
    if makeFuzzy
        outFuzzyFilename = fullfile(outDir, ...
            ['fuzzy_' outTag '.nii.gz']);
    else
        outFuzzyFilename = [];
    end
    
    % Check for the output files
    haveFiles = true;
    for name = {outPetFilename outCtFilename outLabelsFilename}
        haveFiles = haveFiles && exist(name{1}, 'file');
    end
    
    % Skip if the files already exist
    if ~replaceAny && haveFiles
        disp(['Found files for case ' idTag '. Skipping...'])
        continue
    end
    
    % Load the organs
    peaks = load(peaksFilename);
    [organs, ctUnits] = imRead3D(organsFilename);
    petUnits = abs(diag(peaks.matPet2Ct(1:3, 1:3))) .* ctUnits;
    
    % Extract the desired organ
    organ = organs == getOrganLabel(organName);
    clear organs
    if isempty(organ)
        warning(['Organ ' organName ' missing from file ' organsFilename...
            ' skipping this case...'])
        continue
    end
    
    % Read the image volumes
    petFull = imRead3D(petFilename);
    ctFull = imRead3D(ctFilename);
    lesionsFull = imRead3D(labelsFilename);
    
    % Put all the state in a stuct
    opts = struct('ct', ctFull, 'pet', petFull, 'petUnits', petUnits, ...
        'lesions', lesionsFull, 'outPetFilename', outPetFilename, ...
        'outCtFilename', outCtFilename, 'outLabelsFilename', ...
        outLabelsFilename, 'organ', organ, 'lungMode', lungMode, ...
        'replacePet', replacePet, 'replaceCt', replaceCt, ...
        'replaceLabels', replaceLabels, 'res', res, ...
        'matPet2Ct', peaks.matPet2Ct, 'organMasked', organMasked, ...
        'closeMm', closeMm, 'outVoronoiFilename', outVoronoiFilename, ...
        'outFuzzyFilename', outFuzzyFilename);
    
    % Process and save
    processPetCtHelper(opts);
end
