function idx = cellGetStrIdx(c, substr)
% See cellGetStr

idx = find(~cellfun(@isempty, strfind(c, substr)));

end
