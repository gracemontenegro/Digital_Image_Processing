%%
%   Project 1: Intensity Levels
%   Course:    EE485 / CES 540 Digital Data Transmission
%   Professor: Brendan Hamel-Bissell
%   Student:   Grace Montenegro
%   Date:      02/01/2018
%   Description:
%       Given input: Image
%       Variables: Number of Intensity levels in integer powers of 2. 
%       Output: A new image resulted by reducing the number of intensity  
%               levels in the given image.
% ***********************************************************************

close all;
clear

% Read, store and display the image

figure

A = imread('Fig0221a_ctskull-256.tif');
imagesc(A)
title('Original Image')
axis image
axis off
colormap gray


figure

B = A; 

ILevel = 8; % Num. of Intensity levels. It has to be in integer power of 2

% Loop for varying the intensity level of the image

for i = 0:ILevel-1
    B(B>256*i/(ILevel) & B<=256/(ILevel/(i+1))) = (256/ILevel)*(i+1)-1;
    
end

% Displays the new image

imagesc(B)
title('New Image')
axis image
axis off
colormap gray

% display the histogram for the new image
figure
imhist(B)
title('Histogram New Image')

% ************************************************************************

% Explanation:
%
% There are 256 different possible intensities in an 8-bit grayscale image.
% The histogram graphic of an image shows the pixel intensity values, it 
% means, it shows the distribution of pixel through those grayscale values.
% Varying the intensity level, in the histogram graphic we see the changes
% in the distribution of pixel through those grayscale values. 
% As the number of intensity levels increases, the impact in the resulting
% image is more clear, since the resolution gets better, the images is more
% continous.
% In relation to the storage memory for the image (mxn array), intensity  
% levels is equal to 2 to the power of k ( 2^k), which k is the number
% of bits per pixel. The total of bits is then equal to mxnxk. 
% So that, reducing the number of intensity level, reduce the storage
% memory for the image, but, as well, the resolution or quality of the
% image is affected. 
