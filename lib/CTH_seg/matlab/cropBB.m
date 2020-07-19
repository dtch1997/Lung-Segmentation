function crop = cropBB(vol, bb)
% Crop a volume given a bounding box

crop = vol(bb(2) : bb(2) + bb(5) - 1, ...
    bb(1) : bb(1) + bb(4) - 1, ...
    bb(3) : bb(3) + bb(6) - 1);

end