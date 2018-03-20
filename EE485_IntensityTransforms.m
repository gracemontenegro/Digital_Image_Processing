%%
%   Project 2: Intensity Transforms
%   Course:    EE485 / CES 540 Digital Data Transmission
%   Professor: Brendan Hamel-Bissell
%   Student:   Grace Montenegro
%   Date:      02/08/2018
%   Description:
%       Determine the optimal Gamma value to obtain the best possible 
%       visual result for the given image. 
%       Given input: Image
%       Variables: Gamma 
%       Output: Optimized Gamma transformed image  
%               
% ***********************************************************************


close all;
clear

% Read and display the given image and its histogram

B = imread('Fig0309a_washed_out_aerial_image.tif');
subplot(2,2,1);
imagesc(B)
title('Original Image')
axis image
axis off
colormap gray

subplot(2,2,2);
imhist(B)
title('Histogram Original Image')

% Intensity Transform

I = B; 
gamma = 2;
C = 1;
for i = 0:255
   I(I==i)=(i^gamma)/255 ;
end

% Display the Gamma Corrected image and its histogram

subplot(2,2,3);
imagesc(I)
axis image
axis off
colormap gray
title('Image_Gamma_Corrected');

subplot(2,2,4);
imhist(I)
title('Histogram Image_Gamma_Corrected')

% Show some testing results while looking for the optimal Gamma val

figure(2);
T = B; 
gamma = 0.6;
C = 1;
for i = 0:255
   T(T==i)=(i^gamma)/255 ;
end


subplot(2,2,1);
imagesc(T)
axis image
axis off
colormap gray
title('Image_Gamma = 0.6');

subplot(2,2,2);
imhist(T)
title('Histogram Image_Gamma = 0.6')

Ts = B; 
gamma = 1.5;
C = 1;
for i = 0:255
   Ts(Ts==i)=(i^gamma)/255 ;
end


subplot(2,2,3);
imagesc(Ts)
axis image
axis off
colormap gray
title('Image_Gamma = 1.5');

subplot(2,2,4);
imhist(Ts)
title('Histogram Image_Gamma = 1.5')

%********************************************************************
%
%   Explanation
%   Intensity Transformation is an image processing technique. which
%   transform input intensity into output intensity with the objective
%   of enhancement images. 
%   Gamma transformation is one of the basic function of intensity
%   transformation using the formula of s = c*r^gamma.
%   In this assignment, we are looking for the optimal gamma value to be
%   applied to the given image. 
%   The original image looks light, and as we can see in its histogram, the
%   major weights are going to the light side (right). A little of 
%   compresion of the intensity levels needs to be done. 
%   Testing with values < 1, would not a good idea because the image would
%   lost because gets dark, as you can see when gamma = 0.6. 
%   Instead, values of gamma > 1. With gamma = 1.5 and 2.0, the images
%   looks pretty similar, but, observing the histograms of both, with the
%   value of 2.0, it's more balanced. Going higher than 2.0 it becomes too
%   light, losing the image. So, the best enhancement in terms of constrast
%   was obtained with the optimal gamma value equal 2.0. 
%  
%**************************************************************************

% Equalization process

A=0;
for i = 0:255
    A(B == i) = 1;
    Total(i+1) = sum(sum(A));
    A=0;
end    

Sk = (0:255);
[M,N] = size(B);
L = 256;


for k = 1:256
    Sk(k) = (L-1)/(M*N) * sum(Total(1:k));
end

Equalized_img = B;
for j = 1:256
    Equalized_img(B == j) = Sk(j);
end

figure(3)
imagesc(Equalized_img)
title('Equalized Image')
axis image
axis off
colormap gray

%***********************************************************************
% Comments
% With Equalization, the enhancement of the image obtained looks better
% or with a higher contrast than the obtained with the Gamma
% transformation.
% The final image will depend on what are we looking for or what we need
% to see/get from the image.
%***********************************************************************


%%

%*************************************************************************
% This section is using an image that shows 5 squares. With the equalization
% process for example, we could find what it is hidden in each squares.
% ************************************************************************

close all;
clear

% Read and display the given image and its histogram

B = imread('Fig0326a_embedded_square_noisy_512.tif');
figure(4)
subplot(2,2,1);
imagesc(B)
title('Original Image')
axis image
axis off
colormap gray

subplot(2,2,2);
imhist(B)
title('Histogram Original Image')


% Equalization process

A=0;
for i = 0:255
    A(B == i) = 1;
    Total(i+1) = sum(sum(A));
    A=0;
end    

Sk = (0:255);
[M,N] = size(B);
L = 256;


for k = 1:256
    Sk(k) = (L-1)/(M*N) * sum(Total(1:k));
end

Equalized_img = B;
for j = 1:256
    Equalized_img(B == j) = Sk(j);
end

figure(5)
subplot(2,2,1)
imagesc(Equalized_img)
title('Equalized Image')
axis image
axis off
colormap gray

subplot(2,2,2);
imhist(Equalized_img)
title('Histogram Equalized Image')
