dirlist = dir('../train - Copy');

%param.imageSize = [256 256]; % it works also with non-square images
param.orientationsPerScale = [8 8 8 8];
param.numberBlocks = 4;
param.fc_prefilt = 4;

%img1 = imread('demo1.jpg');

gists = zeros(length(dirlist)-2,512);
for i = 3:length(dirlist)
    filename = strcat('../train - Copy/',dirlist(i).name)
    img = imread(filename);
    [gist, param] = LMgist(img, '', param);
    gists(i-2,:) = gist;
    %imshow(gist)
end

csvwrite('flipped_gists.csv',gists)

