%%   
%************************************************************
%   Project 5: Fourier Filtering 
%   Course:    EE485 / CES 540 Digital Data Transmission
%   Professor: Brendan Hamel-Bissell
%   Student:   Grace Montenegro
%   Date:      03/08/2018
%   Goal:      Use image Fourier transforms to process images
% 
%************************************************************

close all;
clear

% Read the given image

A = imread('Fig0425a_translated_rectangle.tif');

% Calculate and display the magnitude of the Fourier transform of the 
% attached translated rectangle in log10 scale. Explain the features you 
% see in the Fourier transform -- what shapes / features in the original 
% image cause these features? 
%
% Fourier transform descomposes a function of time (signal) into the freq. 
% that make it up. So, a sinc wave with a FT results into square waves, or 
% viceversa. In this case, it seems to be the first one.
%**************************************************************************

F = fftshift(fft2(A));  % Fourier shift and Fourier Transform image
Amplitude_A = abs(F);   % Getting the magnitude of the FT image
Angle_A = angle(F);     % Getting the angle of the FT image
logScale = 10*log10(Amplitude_A); % log base 10 image from the magnitude

% plot original and resultant images
figure(1)
subplot(1,2,1);
imagesc(A)
title('Original Image')
axis image
axis off
colormap gray


subplot(1,2,2);
imagesc(logScale)
title('Log10 of magnitude image')
axis image
axis off
colormap gray


%%

% Calculate and display the magnitude of the Fourier transform of the 
% attached translated rectangle in log10 scale, after you have rotated the 
% image by 45 degrees. Explain the impact of rotation on the Fourier 
% transform.
% 
% Conclusion: The impact is similar to the one before but with the effect 
% of the 45 degrees rotation.
%*************************************************************************

B = imrotate(A, 45); % rotate the orginal image by 45 degrees

FB = fftshift(fft2(B));
Amplitude_B = abs(FB);
Angle_B = angle(FB);
logScale_B = log10(Amplitude_B);

figure(2)
subplot(1,2,1);
imagesc(B)
title('Original Image')
axis image
axis off
colormap gray


subplot(1,2,2);
imagesc(logScale_B)
title('Amplitude Image')
axis image
axis off
colormap gray

%%

% Create a low-pass filter, and use the low-pass filter to perform an 
% unsharp, or high-boost, mask on the attached blurry moon picture. 
% Explain how you chose the size of your filter and how you chose the 
% filter shape. Try both a simple step function filter and a Gaussian 
% low-pass filter then explain the difference in your results. 
%
% Conclusions: Using the blurry moon picture and a simple step function and 
% a Gaussian low-pass filter, I got a High-boost sharpenning image. 
% Playing with the constants, for example with the radius, the smaller the 
% rad more sharpenning I got. And the booster constant plays with the 
% constract. we have to recall that low pass filters plays with blurry.
% As we have seen with any image processing, the optimal filtered image,
% would depend on what the final user is looking for. For this case, I
% consider the Gaussian reflect a more realistic image than with the stepF,
% on the other hand, the stepF one could remark things that could be of the 
% interest of the final user. The net result looking for the given image
% is an image in which small details were enhanced and the background 
% tonality was reasonably preserved.
% Aditionally, I have realized that I could come out with a very but very
% similar result in both cases, it is just a matter of play with the
% constants involved. So we could work with any of them. 
%************************************************************************

O = imread('Fig0338a_blurry_moon.tif'); % read image
C = double(O);
[M,N] = size(C);
[x,y] = meshgrid(1:N, 1:M);

% Step Function filter low-pass  

rad = 5;        % radius of the circle to filter

Low_P = ((y - (M/2)).^2 + (x - (N/2)).^2) <= rad.^2 ; % Low Pass filter

FC = fftshift(fft2(C)); % Shift and Fourier Transform of the image
FC = FC .* Low_P;       % FT * Filter
IF = abs(ifft2(FC));    % Magnitude of the inverse FT image
IF = uint8(IF); 
Gmask = O - IF;         % Substract from original the filtered image 
k= 2;                   % Constant for booster 
G = O + k.*Gmask;       % Add a weighted portion of the mask back to the orig  

% plot images
figure(3)
subplot(1,2,1);
imagesc(O)
title('Original Image')
axis image
axis off
colormap gray

subplot(1,2,2);
imagesc(G)
title('LP Filtered image (StepF)')
axis image
axis off
colormap gray

%%
% Gaussian low-pass filter
% The steps are the same, the difference is the filter

[x,y] = meshgrid(1:N, 1:M);
c = 3000;   % slope constant
GF = exp(-(((y - M/2).^2+(x - N/2).^2))/c); % Gaussian Low Pass filter

FT = fftshift(fft2(C)); % Shift Fourier Transform
FT = FT .* GF;          % FT * Filter
IFT = abs(ifft2(FT));   % Amplitude inverse FT
IFT = uint8(IFT);
Gmask_g = O - IFT;    % Substract from original the filtered image
k= 2;                 % Constant for booster
GC = O + k.*Gmask_g;  % Add a weighted portion of the mask back to the orig 

figure(4)
subplot(1,2,1);
imagesc(O)
title('Original Image')
axis image
axis off
colormap gray

subplot(1,2,2);
imagesc(GC)
title('Filtered image (Gaussian LP)')
axis image
axis off
colormap gray

% display both filtered images for comparison 
figure(5)
imshowpair(G, GC, 'montage');

%%
%**************************************************************************
% Create a high-pass filter, and use it to find the edges of the attached 
% thumb print picture. Use either a step function or 1-Gaussian to select 
% only the high frequencies. Once your filter is implemented, use 
% thresholding to force all values in the image to either 255 or 0. Your 
% final image should clearly show the ridges (black) and valleys (white). 
% This is how Apple pre-processes their fingerprint images before the 
% algorithm that matches them to unlock your phone. 
%
%************************************************************************
% Conclusions: I used the step function and select only the high frequencies
% and use thresholding as it was suggested. The resultant filtered image
% it is showing the ridges and valleys. For this exercises, the instructions
% were acomplished. For further studies if there were time, there are two 
% areas that it would need to work even more on it, in order to get the 
% the filtered image even better. The way to do this, it would be using the 
% histogram to localize the specifics areas and this way apply as well the 
% corresponding thresholding. 
%************************************************************************

O = imread('Fig0457a_thumb_print.tif'); % read image
C = double(O);
[M,N] = size(C);
[x,y] = meshgrid(1:N, 1:M);

% high-pass filter

rad = 5;

High_P = ((y - (M/2)).^2 + (x - (N/2)).^2) >= rad.^2 ;

FC = fftshift(fft2(C));     % Shift Fourier Transform
FC = FC .* High_P;          % FT * Filter
IF = abs(ifft2(FC));        % Magnitude of inverse FT image

GHmask = double(O) - IF;    % Substract from original the filtered image
k= 2;
GH = double(O) + k.*GHmask; % Add back to orig a weighted portion of the mask 

GH = GH - min(min(GH));       % Normalizing
GH = GH/max(max(GH))*255; 
GH = uint8(GH);

% binarizing image
T = 140;

GH(GH<=T) = 0;  
GH(GH>T) = 255; 

% plot images
figure(6)
subplot(1,2,1);
imagesc(O)
title('Original Image')
axis image
axis off
colormap gray

subplot(1,2,2);
imagesc(GH)
title('HP Filtered image')
axis image
axis off
colormap gray

%%

%********************** PART II ******************************************

% Calculate the Fourier transform of the attached car image. Identify the 
% features in the Fourier transform that cause the Moir? pattern in the 
% image.
%
% COMMENTS ABOUT RESULTS: First of all, we have a newspaper image (given).  
% The sampling lattice (vertical and horizontal) and dot patterns oriented 
% at +- 45 degrees interact to create a uniform moire pattern, so the image
% looks blotchy. 
% In the spectrum image, the features that are causing the moire pattern
% are the white crosses shown. 

% Read the given image
A = imread('Fig0464a_car_75DPI_Moire.tif');

% Calculating the Fourier transform of the given image

F = fftshift(fft2(A));  % Fourier shift and Fourier Transform image
Amplitude_A = abs(F);   % Getting the magnitude of the FT image
logScale = 10*log10(Amplitude_A); % log base 10 image from the magnitude

% plot original and resultant images
figure(7)
subplot(1,2,1);
imagesc(A)
title('Original Image')
axis image
axis off
colormap gray


subplot(1,2,2);
imagesc(logScale)
title('Log10 of magnitude image')
axis image
axis off
colormap gray

%%
%************************************************************************
% Create 5 different Gaussian bandpass filters at different cutoff freq, 
% and use these filters to create 5 images of the attached car image. 
% Each filtered image should show the components of the original image at 
% the frequencies allowed by the bandpass filter. The idea here is to 
% separate the image into frequency bins and look at each one individually. 
% The lowest frequency filter could be a low-pass, and the highest could be 
% a high pass, but make sure these are sized appropriately to illustrate 
% the different parts of the image at in each frequency band. Also, you'll 
% want to design the bandpass filters to select the features you identified 
% above to verify they are the cause of the pattern. 

% Read the given image
O = imread('Fig0464a_car_75DPI_Moire.tif');

C = double(O);
[M,N] = size(C);
[x,y] = meshgrid(1:N, 1:M);

Distance = sqrt((y - (M/2)).^2 + (x - (N/2)).^2);

% Filter No 1
% A high width of the band is used with a low cutoff frequencies to get a
% desired image

w = 35;
cutoff_f = 7;

GF = exp(-(((((Distance).^2) - ((cutoff_f).^2)))./(Distance.*w)).^2); % Gaussian

FC = fftshift(fft2(C));     % Shift Fourier Transform
logScale = log10(abs(FC));
FC = FC .* GF;              % FT * Filter
IF = abs(ifft2(FC));        % Magnitude of inverse FT image


% plot original and resultant images
figure(8)
subplot(2,2,1);
imagesc(O)
title('Original Image')
axis image
axis off
colormap gray

subplot(2,2,2);
imagesc(IF)
title('Gaussian BandPass image')
axis image
axis off
colormap gray

%figure(9)
subplot(2,2,3);
imshowpair(logScale, GF,'blend');
title('overlapping Filter and LogScale image')
axis image
axis off
colormap gray

%%
% For the following filters, I am using the same width of the band but 
% changing the the cutoff frequencies. We will see the band pass for each
% and we could see filtering low frequencies and including the white stars
% that cause the moire pattern. For demo purpose and have a better understanding 
% we could see in each how impact the image depending on how many of the 
% white stars are included. It looks dark because the low frequecies are
% supreme, and the edges shown are due the high frequencies included.


% Filter No 2

w = 20;
cutoff_f = 30;

GF = exp(-(((((Distance).^2) - ((cutoff_f).^2)))./(Distance.*w)).^2); % Gaussian

FC = fftshift(fft2(C));     % Shift Fourier Transform
logScale = log10(abs(FC));
FC = FC .* GF;              % FT * Filter
IF = abs(ifft2(FC));        % Magnitude of inverse FT image


% plot original and resultant images
figure(10)
subplot(2,2,1);
imagesc(O)
title('Original Image')
axis image
axis off
colormap gray

subplot(2,2,2);
imagesc(IF)
title('Gaussian BandPass image')
axis image
axis off
colormap gray

%figure(11)
subplot(2,2,3);
imshowpair(logScale, GF,'blend');
title('overlapping Filter and LogScale image')
axis image
axis off
colormap gray

%%
% Filter No. 3

w = 20;
cutoff_f = 60;

GF = exp(-(((((Distance).^2) - ((cutoff_f).^2)))./(Distance.*w)).^2); % Gaussian

FC = fftshift(fft2(C));     % Shift Fourier Transform
logScale = log10(abs(FC));
FC = FC .* GF;              % FT * Filter
IF = abs(ifft2(FC));        % Magnitude of inverse FT image


% plot original and resultant images
figure(12)
subplot(2,2,1);
imagesc(O)
title('Original Image')
axis image
axis off
colormap gray


subplot(2,2,2);
imagesc(IF)
title('Gaussian BandPass image')
axis image
axis off
colormap gray

%figure(13)
subplot(2,2,3);
imshowpair(logScale, GF,'blend');
title('overlapping Filter and LogScale image')
axis image
axis off
colormap gray

%%
% Filter No. 4

w = 20;
cutoff_f = 90;

GF = exp(-(((((Distance).^2) - ((cutoff_f).^2)))./(Distance.*w)).^2); % Gaussian

FC = fftshift(fft2(C));     % Shift Fourier Transform
logScale = log10(abs(FC));
FC = FC .* GF;              % FT * Filter
IF = abs(ifft2(FC));        % Magnitude of inverse FT image


% plot original and resultant images
figure(14)
subplot(2,2,1);
imagesc(O)
title('Original Image')
axis image
axis off
colormap gray


subplot(2,2,2);
imagesc(IF)
title('Gaussian BandPass image')
axis image
axis off
colormap gray

%figure(15)
subplot(2,2,3);
imshowpair(logScale, GF,'blend');
title('overlapping Filter and LogScale image')
axis image
axis off
colormap gray

%%
% Filter No. 5

w = 20;
cutoff_f = 120;

GF = exp(-(((((Distance).^2) - ((cutoff_f).^2)))./(Distance.*w)).^2); % Gaussian

FC = fftshift(fft2(C));     % Shift Fourier Transform
logScale = log10(abs(FC));
FC = FC .* GF;              % FT * Filter
IF = abs(ifft2(FC));        % Magnitude of inverse FT image


% plot original and resultant images
figure(16)
subplot(2,2,1);
imagesc(O)
title('Original Image')
axis image
axis off
colormap gray


subplot(2,2,2);
imagesc(IF)
title('Gaussian BandPass image')
axis image
axis off
colormap gray

%figure(17)
subplot(2,2,3);
imshowpair(logScale, GF,'blend');
title('overlapping Filter and LogScale image')
axis image
axis off
colormap gray

%%

% Gaussian low-pass filter

% In this case, the filter exclude the white stars to try to eliminate the
% moire pattern. A low-pass filter plays with the blurry. My eyes are not
% the best, and as all the cases, the ideal final image it depends on the
% final users needs, it is subjective. Changing the c constant, we could
% make it blurrier or less blurrier, depends on the users eyes at the end
% how they see it better.

O = imread('Fig0464a_car_75DPI_Moire.tif');
C = double(O);
[M,N] = size(C);
C = uint8(C);

[x,y] = meshgrid(1:N, 1:M);
c = 1000;   % slope constant
GF = exp(-(((y - M/2).^2+(x - N/2).^2))/c); % Gaussian Low Pass filter

FT = fftshift(fft2(C)); % Shift Fourier Transform
logScale = log10(abs(FT));
FT = FT .* GF;          % FT * Filter
IFT = abs(ifft2(FT));   % Amplitude inverse FT


figure(18)
subplot(2,2,1);
imagesc(O)
title('Original Image')
axis image
axis off
colormap gray

subplot(2,2,2);
imagesc(IFT)
title('Filtered image (Gaussian LP)')
axis image
axis off
colormap gray

%figure(19)
subplot(2,2,3);
imshowpair(logScale, GF,'blend');
axis image
axis off
colormap gray

%%

% Gaussian BandPass Filter

% This one, actually was taking from internet, it shows another way to work
% with Gaussian BandPass Filter, which it is the result of the
% multiplication of a GLowPass filter with the GHigh pass filter. The
% filter looks good but the image that results it does not seems consistent
% or as I would expect. So I will go more with the method learned in the
% class.

O = imread('Fig0464a_car_75DPI_Moire.tif');
C = double(O);
[M,N] = size(C);
C = uint8(C);

FC = fftshift(fft2(C,2*M-1,2*N-1));     % Shift Fourier Transform
logScale = log10(abs(FC));

figure(32)
subplot(2,2,1)
imagesc(C);
title('Original Img')
axis image
axis off
colormap gray

subplot(2,2,2)
imagesc(logScale)
title('Fourier Spectrum of Img')
axis image
axis off
colormap gray

% Initialize filters and defining cutoff frequencies
Hlp = ones(2*M-1,2*N-1);
Hhp = ones(2*M-1,2*N-1);
Hbp = ones(2*M-1,2*N-1);
dh = 10;
dl = 100;

for m = 1:2*M-1
    for n =1:2*N-1
        dist = sqrt((m-(M+1))^2 + (n-(N+1))^2);
        Hlp(m,n) = exp(-dist^2/(2*dl^2));   % Gaussian LowPass Filter
        Hhp(m,n) = 1.0 - exp(-dist^2/(2*dh^2)); % Gaussian HighPass Filter
        Hbp(m,n) = Hlp(m,n).*Hhp(m,n); % Gaussian BandPass Filter
    end
end


subplot(2,2,3)
imagesc(Hbp)
title('Freq Domain Filter Fn Img')
axis image
axis off
colormap gray

FI = FC + Hbp.*FC;
%FI = FC .* Hbp;
FI = ifftshift(FI);
FI = ifft2(FI, 2*M-1, 2*N-1);
FI = real(FI(1:M,1:N));
subplot(2,2,4)
imagesc(FI)
title('Bandpass Filtered Img')
axis image
axis off
colormap gray


%%
% Create a notch filter (you'll need 8 notches) to eliminate the features 
% that cause the Moire pattern. Compare this approach to an approach using
% 2 bandreject filters. Which results in better image quality? 
%
% The 8 notch filters were created. Using symmetric around the center, I
% displayed the filters and correspondig image when I combined two of them.
% It was interesting to see the two of the pairs, specifically the filters
% closer to the center, some features that causes the Moire pattern were
% eliminate but you can see that the ones that stays are going in the same
% direction of the filters 

% Read the given image
O = imread('Fig0464a_car_75DPI_Moire.tif');

C = double(O);
[M,N] = size(C);
[x,y] = meshgrid(1:N, 1:M);
X = 56;
Y = 42;
Distance = sqrt((y - Y).^2 + (x - X).^2);

% Notch No.1

Sig = 20;

HN1 = 1 - exp(-((Distance).^2./(Sig).^2)); % Notch Filter

FC = fftshift(fft2(C));     % Shift Fourier Transform
logScale = log10(abs(FC));
FC = FC .* HN1;             % FT * Filter
IF = abs(ifft2(FC));        % Magnitude of inverse FT image

% Notch No.2

X = 55;
Y = 85;
Distance = sqrt((y - Y).^2 + (x - X).^2);

Sig = 20;

HN2 = 1 - exp(-((Distance).^2./(Sig).^2)); % Notch Filter

FC = fftshift(fft2(C));     % Shift Fourier Transform
logScale = log10(abs(FC));
FC = FC .* HN2;             % FT * Filter
IF = abs(ifft2(FC));        % Magnitude of inverse FT image

% Notch No.3

X = 58;
Y = 166;
Distance = sqrt((y - Y).^2 + (x - X).^2);

Sig = 20;

HN3 = 1 - exp(-((Distance).^2./(Sig).^2)); % Notch Filter

FC = fftshift(fft2(C));     % Shift Fourier Transform
logScale = log10(abs(FC));
FC = FC .* HN3;             % FT * Filter
IF = abs(ifft2(FC));        % Magnitude of inverse FT image


% Notch No.4

X = 57;
Y = 206;
Distance = sqrt((y - Y).^2 + (x - X).^2);

Sig =20;

HN4 = 1 - exp(-((Distance).^2./(Sig).^2)); % Notch Filter

FC = fftshift(fft2(C));     % Shift Fourier Transform
logScale = log10(abs(FC));
FC = FC .* HN4;             % FT * Filter
IF = abs(ifft2(FC));        % Magnitude of inverse FT image

% Notch No.5

X = 113;
Y = 41;
Distance = sqrt((y - Y).^2 + (x - X).^2);

Sig = 20;

HN5a = 1 - exp(-((Distance).^2./(Sig).^2)); % Notch Filter
HN5 = HN5a .* HN4;          % Combining Notch symmetrics to the center
FC = fftshift(fft2(C));     % Shift Fourier Transform
logScale = log10(abs(FC));
FC = FC .* HN5;             % FT * Filter
IF = abs(ifft2(FC));        % Magnitude of inverse FT image
%IF = double(O) - IF;

% plot original and resultant images plus filter


figure(20)
subplot(2,2,1);
imagesc(O)
title('Original Image')
axis image
axis off
colormap gray

subplot(2,2,2);
imagesc(IF)
title('Filtered Notch5&4 image')
axis image
axis off
colormap gray

%figure(21)
subplot(2,2,3);
imshowpair(logScale, HN5,'blend');
title('overlapping Filter and LogScale image')
axis image
axis off
colormap gray

%%
% Notch No.6

X = 112;
Y = 82;
Distance = sqrt((y - Y).^2 + (x - X).^2);

Sig = 20;

HN6a = 1 - exp(-((Distance).^2./(Sig).^2)); % Notch Filter
HN6 = HN6a .* HN3;          % Combining Notch symmetrics to the center
FC = fftshift(fft2(C));     % Shift Fourier Transform
logScale = log10(abs(FC));
FC = FC .* HN6;             % FT * Filter
IF = abs(ifft2(FC));        % Magnitude of inverse FT image
%IF = double(O) - IF;

% plot original and resultant images, plus filter
figure(22)
subplot(2,2,1);
imagesc(O)
title('Original Image')
axis image
axis off
colormap gray

subplot(2,2,2);
imagesc(IF)
title('Filtered Notch6&3 image')
axis image
axis off
colormap gray

%figure(23)
subplot(2,2,3);
imshowpair(logScale, HN6,'blend');
title('overlapping Filter and LogScale image')
axis image
axis off
colormap gray


%%
% Notch No.7

X = 114;
Y = 163;
Distance = sqrt((y - Y).^2 + (x - X).^2);

Sig = 20;

HN7a = 1 - exp(-((Distance).^2./(Sig).^2)); % Notch Filter
HN7 = HN7a .* HN2;          % Combining Notch symmetrics to the center
FC = fftshift(fft2(C));     % Shift Fourier Transform
logScale = log10(abs(FC));
FC = FC .* HN7;             % FT * Filter
IF = abs(ifft2(FC));        % Magnitude of inverse FT image
%IF = double(O) - IF;

% plot original and resultant images
figure(24)
subplot(2,2,1);
imagesc(O)
title('Original Image')
axis image
axis off
colormap gray

subplot(2,2,2);
imagesc(IF)
title('Filtered Notch7&2 image')
axis image
axis off
colormap gray

%figure(25)
subplot(2,2,3);
imshowpair(logScale, HN7,'blend');
title('overlapping Filter and LogScale image')
axis image
axis off
colormap gray

%%
% Notch No.8

X = 114;
Y = 203;
Distance = sqrt((y - Y).^2 + (x - X).^2);

Sig = 20;

HN8a = 1 - exp(-((Distance).^2./(Sig).^2)); % Notch Filter
HN8 = HN8a .* HN1;          % Combining Notch symmetrics to the center
FC = fftshift(fft2(C));     % Shift Fourier Transform
logScale = log10(abs(FC));
FC = FC .* HN8;             % FT * Filter
IF = abs(ifft2(FC));        % Magnitude of inverse FT image
%IF = double(O) - IF;

% plot original and resultant images, plus filter
figure(26)
subplot(2,2,1);
imagesc(O)
title('Original Image')
axis image
axis off
colormap gray

subplot(2,2,2);
imagesc(IF)
title('Filtered Notch8&1 image')
axis image
axis off
colormap gray

%figure(27)
subplot(2,2,3);
imshowpair(logScale, HN8,'blend')
title('overlapping Filter and LogScale image')
axis image
axis off
colormap gray


%%
% Combining the 8 Notches

FT = fftshift(fft2(C));     % Shift Fourier Transform
logScale = log10(abs(FT));
HNT = HN1 .*HN2 .*HN3 .*HN4 .*HN5a .*HN6a .*HN7a .*HN8a;
FT = FT .* HNT;             % FT * Filter
IF = abs(ifft2(FT));        % Magnitude of inverse FT image
%IF = double(O) - IF;

% plot original and resultant images, plus filter
figure(28)
subplot(2,2,1);
imagesc(O)
title('Original Image')
axis image
axis off
colormap gray

subplot(2,2,2);
imagesc(IF)
title('Filtered Notch image')
axis image
axis off
colormap gray

%figure(29)
subplot(2,2,3);
imshowpair(logScale, HNT,'blend')
title('overlapping Filter and LogScale image')
axis image
axis off
colormap gray

%%
%******************************
% 2 Gaussian BandReject Filters
%******************************

% Read the given image
O = imread('Fig0464a_car_75DPI_Moire.tif');

C = double(O);
[M,N] = size(C);
[x,y] = meshgrid(1:N, 1:M);


Distance = sqrt((y - (M/2)).^2 + (x - (N/2)).^2);

% first BR filter
w = 25;
cutoff_f = 50;

GF_1 = 1 - exp(-(((((Distance).^2) - ((cutoff_f).^2)))./(Distance.*w)).^2); % Gaussian

% second BR filter
w = 50;
cutoff_f =100;

GF_2 = 1 - exp(-(((((Distance).^2) - ((cutoff_f).^2)))./(Distance.*w)).^2); % Gaussian

% combining both filters
GF = GF_1 .* GF_2;
FC = fftshift(fft2(C));     % Shift Fourier Transform
logScale = log10(abs(FC));
FC = FC .* GF;              % FT * Filter
IF = abs(ifft2(FC));        % Magnitude of inverse FT image


% plot original and resultant images, plus filter
figure(30)
subplot(2,2,1);
imagesc(O)
title('Original Image')
axis image
axis off
colormap gray

subplot(2,2,2);
imagesc(IF)
title('BandReject image')
axis image
axis off
colormap gray

%figure(31)
subplot(2,2,3);
imshowpair(logScale, GF,'blend')
title('overlapping Filter and LogScale image')
axis image
axis off
colormap gray

%%
% Explain the general concept of Fourier filtering, and discuss how the 
% filters change the original image in the frequency spectrum: 
% Fourier filtering generally speaking is doing filtering in the frequency 
% domain, performing fourier transforms and filter multiply, It is computationally
% faster to do it this way than to perform a convolution in the image in 
% spatial domain.
% The cutoff frequency and the width of the band in each of the 5 scenarios  
% shown above, we could see the spectrum filter images in each, how it is 
% when the filtered images when the filter includes the "white stars".
% which are the cause of the moire pattern. As well, we have to recall
% what does low frequencies (eg.intensity) and high frequencies (eg. edges)
% impact images. 
% Using Gaussian Low Pass filter and Gaussian BandPass filter we got an 
% enhance image. The first one got a better result in relation with the 
% moire pattern effect which was decrease. 




