addpath(genpath('~/bbrister/lung_segmentation/lib/aimutil'))
addpath(genpath('~/bbrister/lung_segmentation/lib/cth_seg'))

rootDir = '/data/lidc';
dataDirName = 'LIDC-NIFTI/debug';
dataDir = fullfile(rootDir, dataDirName);

res = 1;

destDir = fullfile(rootDir, ['nodules-' num2str(res) 'mm']);
if ~exist(destDir, 'dir')
    mkdir(destDir)
end

wd = pwd;
cd(dataDir)
volFiles = sort(strsplit(ls('*volume*.nii.gz')))
cd(wd)

if isempty(volFiles{1})
    volFiles = volFiles(2:end);
end

for i = 1 : length(volFiles)
    volName = volFiles{i};
    maskName = regexprep(volName, 'volume', 'segmentation');
    maskPath = fullfile(dataDir, maskName);
    volPath = fullfile(dataDir, volName);
    if ~exist(maskPath, 'file')
                                warning(['Cannot find file ' maskPath ' skipping this case...'])
                                        continue
                                            end

