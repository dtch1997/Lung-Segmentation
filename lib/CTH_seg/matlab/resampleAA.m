% Resample a volume and mask, with anti-aliasing. units gives the current
% units, res gives the target isotropic resolution. Pass an empty mask to
% skip processing the mask.
function volProc = resampleAA(vol, units, res, antiAliasSlope)

    % Default arguments
    if nargin < 4
        antiAliasSlope = [];
    end

    % Verify inputs
    assert(length(units) == 3)
    assert(ndims(vol) == 3 || ndims(vol) == 4)
    
    % Check the trivial case
    if all(units == res)
        volProc = vol;
        return
    end
    
    % Get the interpolation factors and target dimensions
    factors = res ./ units;
    siz = ceil([size(vol, 1) size(vol, 2) size(vol, 3)] ./ factors');
    A = [diag(factors) zeros(3, 1)];
    numChannels = size(vol, 4);
    
    % Warp w/ AA
    volProc = zeros([siz numChannels]);
    for c = 1 : numChannels
        volProc(:, :, :, c) = warpAA(A, vol(:, :, :, c), siz, ...
            antiAliasSlope);
    end
end