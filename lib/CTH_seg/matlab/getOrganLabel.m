function label = getOrganLabel(name)
    
table = organLabels;
names = table(:, 1);
label = table{cellGetStrIdx(names, name), 2};

end