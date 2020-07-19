function warped = warpAA(A, im, siz, antiAliasSlope)
% Warp an image using the provided affine matrix and output size. Performs
% Gaussian pre-smoothing and then cubic interpolation.

addpath(genpath('/home/blaine/aimutil'))

% Default arguments
if nargin < 4 || isempty(antiAliasSlope)
    antiAliasSlope = 1 / 3;
end

% Compute the anti-aliasing factors for each dimension
factors = diag(A);
sigmas = max(factors - 1, 0) * antiAliasSlope;

% Perform Gaussian anti-aliasing
if any(sigmas > 0)
    im = imGaussFilter3D(im, sigmas);
end

warped = invAffine3D(A, im, siz, 'cubic');

end
