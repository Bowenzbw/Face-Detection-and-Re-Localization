
% %%SVF=pi/(2*n)*sin(pi*(2i-1)/(2*n))*(pi/ti);
% SVF = 0.6;
% f = 0.3;
% S = 
% B = 
% % B=(24/pi)*s*(1+0.033*cos(2*pi*d/365))*(cos(phi)*cos(delta)*sin(ws)+ws*sin(phi)*sin(delta));
% % S=B*(1-rho);
% UHI=(2-SVF-f)*(((S*(DTR^3))/U)^(1/4));
data=imread('f.png');
a=rgb2gray(data);
imshow(a);
D_rgb=uint8(data);
C =makecform('srgb2lab'); %??????
I_lab= applycform(D_rgb, C);
ab =double(I_lab(:,:,2:3)); %??lab???a???b??
nrows= size(ab,1);
ncols= size(ab,2);
ab =reshape(ab,nrows*ncols,2);
nColors= 3; %????????4
[cluster_idx,cluster_center] =kmeans(ab,nColors,'distance','sqEuclidean','Replicates',2); %????2?
pixel_labels= reshape(cluster_idx,nrows,ncols);
%??????????
segmented_images= cell(1,3);
rgb_label= repmat(pixel_labels,[1 1 3]);
for k= 1:nColors
color = data;
color(rgb_label ~= k) = 0;
segmented_images{k} = color;
end
figure(),imshow(segmented_images{1}),title('????——??1');
figure(),imshow(segmented_images{2}),title('????——??2');
figure(),imshow(segmented_images{3}),title('????——??3');
figure(),imshow(segmented_images{4}),title('????——??4');%??????????????
m=uint8(rgb_label);