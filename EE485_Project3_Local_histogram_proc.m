%%
%   Project 3: Local Histogram Procesing 
%   Course:    EE485 / CES 540 Digital Data Transmission
%   Professor: Brendan Hamel-Bissell
%   Student:   Grace Montenegro
%   Date:      02/15/2018
%   Description:
%       Use local histogram processing to recover the data in the attached 
%       image. What information is contained in the dark squares? 
%       Given input: Image 
%       Output: Enhanced image.  
%               
% ***********************************************************************


close all;
clear

% Read the given image

A = imread('Fig0326a_embedded_square_noisy_512.tif');
%A= imread('Fig0309(a)(washed_out_aerial_image).tif');
%A= imread('Fig0316(4)(bottom_left).tif');
% Setting variables

[M,N] = size(A);
[map, intensity_levels] = imhist(A);
L = max(intensity_levels);

B = A;
w = 21; % window

% Calculate global mean and variance

mean_t = 1 / (M * N) * sum(sum(A));

variance = 1 / (M * N) * sum(sum(power((A-mean_t),2)));

%************************************************************************
% Histogram manipulation can be used for image enhancement. In this project
% it will be applied to a given image different method: Global
% Equalization, Local Equalization, Statistics and a combination of Local
% Equalization with Statistics process. 
% Each approach has its advantages, and the optimal process is determined
% by the characteristics of the input and the necessity or use of the
% output. For practical purpose in this project, it was defined a windows
% size and constants values that applied the best possible for each
% process. For instance, the window defined was 21, since smaller like 3, it
% was not the best for the local equalization since the hidden figures were 
% not shown. In all cases the content in the squares were revealed, the 
% differ in the contrast, but again, any of them could be the best
% depending on the application to be used. Depending on the characteristic
% of the input, one method or another is more useful, like cases with MRI
% images or landscapes pictures, the enhancement required is relative to
% the application or needs. 
%************************************************************************

%%

%***********************************
%   Image Statistics process
%***********************************
% The Mean and the variance are used for enhancement purposes. The global
% mean and variance are applied over the entire image, which helps for
% gross adjustments in overall intensity and contrast. As well, the mean
% and variance of the window (local) are useful for make adjustments in the
% neighborhood about each pixel in the image. The constant values will
% depend on the image. In this case, we are looking for possible hidden
% figures in the black squares, so it is neccesary enhance the dark areas.
% A comparison of the local mean vs. the global mean with a factor, needs 
% to be done. Also, the variances are used for evaluate enhancing areas
% that have low contrast. For those areas that meet the conditions for
% enhancement, a constant (E) is multiplied it in order to increase or
% decrease the value of its intensitive level relative to the rest of the
% image.
% As result of this process, the hidden figures from the dark squares were
% shown. 
%*************************************************************************

% Setting constants

c = 0.8;
c1 = 0.07;
c2 = 0.7;
E = 20;
B = A;
for m = 1:M-w
    for n = 1:N-w
        sub = A(m:m+w, n:n+w);
        means = mean(mean(sub));
        variance_s = var(var(double(sub))); 
        if ((means <= c*mean_t) & (c1*variance <= variance_s <= c2*variance))
            
            B(m:m+w, n:n+w) = E.*sub;
            
        elseif ((means > c*mean_t)  & (c1*variance > variance_s > c2*variance))
            
            B(m:m+w, n:n+w) = sub;
        end    
    end
end

figure(1)
subplot(2,2,1);
imagesc(A)
title('Original Image')
axis image
axis off
colormap gray

subplot(2,2,2);
imhist(A)
title('Histogram Original Image')
colormap gray

subplot(2,2,3);
imagesc(B)
title('Image Statistics')
axis image
axis off
colormap gray

subplot(2,2,4);
imhist(B)
title('Histogram Image Statistics')
colormap gray

%%

%***********************************
%   Global Equalization process
%***********************************
% With this process, pixels are modified by a transformation function -
% using the given formulas for it - based on the intensity distribution of
% the entire image, obtaining an overall enhancement. As a result of this
% process, the hidden images were shown, all the light areas turned more
% gray, this can be seen as well in the histogram. 
%*************************************************************************

Equalized_img = double(A);
n_p=0;
for i = 0:L
    n_p(A == i) = 1;
    Total(i+1) = sum(sum(n_p));
    n_p=0;
end    

Sk = (0:L);

for k = 1:L+1
    Sk(k) = L/(M*N) * sum(Total(1:k));
end

for j = 1:L+1
    Equalized_img(A == j) = Sk(j);
end

figure(4)
subplot(2,2,1);
imagesc(A)
title('Original Image')
axis image
axis off
colormap gray

subplot(2,2,2);
imhist(A)
title('Histogram Original Image')
colormap gray

subplot(2,2,3);
imagesc(Equalized_img)
title('(Global) Equalized Image')
axis image
axis off
colormap gray

subplot(2,2,4);
Equalized_img = uint8(Equalized_img);
imhist(Equalized_img)
title('Histogram Global Equalization')
colormap gray


%%

%***********************************
%   Local Equalization process
%***********************************
% This process works on the intensity distribution in a neighborhood of
% every pixel in the image. A window is defined for it, and moves its
% center from pixel to pixel, mapping the intensity of the pixel centered
% in the neighborhood, repeating the process for each window until the
% entire image is covered. This method or approach is used when it is
% necessary to enhance details over small areas in an image.
% The output reveals a more detailed image. The histogram of the equalized
% image span a wider range of the intensity scale.
%*************************************************************************

Equalized_img = double(A);

for m = 1:M-(w - 1)
    for n = 1:N-(w-1)
        sub = A(m:m+(w-1), n:n+(w-1));
        [fil,col] = size(sub);
        
        C=0;
        for i = 0:L
            C(sub == i) = 1;
            Total(i+1) = sum(C);
            C=0;
        end
        
        Sk = (0:L);
        for k = 1:L+1
            Sk(k) = L/(fil*col) * sum(Total(1:k));
        end
        
        middle = A((((2*m)+w-1)/2),(((2*n)+w-1)/2));
        Equalized_img((((2*m)+w-1)/2),(((2*n)+w-1)/2)) = Sk(middle+1);
    end
end
figure(2)
subplot(2,2,1);
imagesc(A)
title('Original Image')
axis image
axis off
colormap gray

subplot(2,2,2);
imhist(A)
title('Histogram Original Image')
colormap gray

subplot(2,2,3);
imagesc(Equalized_img)
title('(Local) Equalized Image')
axis image
axis off
colormap gray

Equalized_img = uint8(Equalized_img);
subplot(2,2,4);
imhist(Equalized_img)
title('Histogram Local Equalization')


%%

%***********************************
%   Local Equalization combined 
%     with Statistics process
%**********************************
% In this case, it was used a combination of processes, using the same 
% constants values used before individually, the output shows different
% details in the border of each squares, as well, the histogram span
% narrower range of the intensity scale vs the Local Equalization without
% Statistics, although span a wider range than the original. 
%*************************************************************************


c = 1.8;
c1 = 1.07;
c2 = 1.7;
E = 20;
Equalized_img = double(A);

for m = 1:M-(w - 1)
    for n = 1:N-(w-1)
        sub = A(m:m+(w-1), n:n+(w-1));
        means = mean(mean(sub));
        variance_s = var(var(double(sub))); 
        [fil,col] = size(sub);
        
        n_p=0;
        for i = 0:L
            n_p(sub == i) = 1;
            Total(i+1) = sum(n_p);
            n_p=0;
        end
        
        Sk = (0:L);
        for k = 1:L+1
            Sk(k) = L/(fil*col) * sum(Total(1:k));
        end

        middle = A((((2*m)+w-1)/2),(((2*n)+w-1)/2));
        
        if ((means <= c*mean_t) & (c1*variance <= variance_s <= c2*variance))
            
            Equalized_img(m:m+(w-1), n:n+(w-1)) = Sk(middle+1);
            
        elseif ((means <= c*mean_t) & (c1*variance <= variance_s <= c2*variance)) 
            
            Equalized_img(m:m+(w-1), n:n+(w-1)) = sub;
        end    
        
        
        
    end
end

figure(3)
subplot(2,2,1);
imagesc(A)
title('Original Image')
axis image
axis off
colormap gray

subplot(2,2,2);
imhist(A)
title('Histogram Original Image')
colormap gray

subplot(2,2,3);
imagesc(Equalized_img)
title('Local/Statistic Equalized Image')
axis image
axis off
colormap gray

Equalized_img = uint8(Equalized_img);
subplot(2,2,4);
imhist(Equalized_img)
title('Histogram Equalized Image')
colormap gray

