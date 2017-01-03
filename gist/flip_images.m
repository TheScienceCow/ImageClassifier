dirlist = dir('../train - Copy');
for i = 3:length(dirlist)
    filename = strcat('../train - Copy/',dirlist(i).name)
    img = imread(filename);
    imwrite(flip(img,2),filename)
    %imshow(gist)
end