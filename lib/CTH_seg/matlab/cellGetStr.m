    function str = cellGetStr(c, substr)
% Return the strings in a cellstring containing the given substring
    str = c{cellGetStrIdx(c, substr)};
end