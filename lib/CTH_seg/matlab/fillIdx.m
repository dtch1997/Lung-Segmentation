function vol = fillIdx(vol, subVol, idx)
% Fill in a sub-volume given the starting indices
%
% See also:
%   cropBB

vol(idx(1) : idx(1) + size(subVol, 1) - 1, ...
    idx(2) : idx(2) + size(subVol, 2) - 1, ...
    idx(3) : idx(3) + size(subVol, 3) - 1) = subVol;

end