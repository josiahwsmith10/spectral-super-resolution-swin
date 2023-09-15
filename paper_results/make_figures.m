%% Add util functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath(genpath("matlab_util"))

%% Experiment 1: PSNR / SSIM across SNR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[t_PSNR,t_SSIM] = exp1_table();
[f1,h1] = exp1_figures();

%% Experiment 2: resolution capability
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
t_sep = exp2_table();
[f2,h2] = exp2_figure();

%% Experiment 3: comparison of sidelobes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[f3,h3] = exp3_figures();

%% Experiment 4: spinning point scatterers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[f4,h4] = exp4_figures();

%% Experiment 5: real Boeing data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[f5,h5] = exp5_figures();

%% Shifted Window Example
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[fswin,hswin] = shifted_window_example();

%%
load paper_data/plane.mat

Y = zeros(300,4096);
for i = 1:300
    Y(i,:) = fft(x(i,:),4096);
end

%%
imagesc(abs(fftshift(ifft(Y.',512,2))))

%%
imagesc(abs(fftshift(ifft(cresfreq.',512,2))))

%%
load paper_data/B727r.mat

clf
A = fftshift(fft(x,4096,2),2);

A = flip(flip(A,1),2);

A = 20*log10(abs(A/max(A(:))));
A(A<-40) = -40;

h = handle(axes());
mesh(A,"FaceColor","interp","EdgeColor","none")

xlabel("Relative Range / m");
view(h,2)

ylabel("Pulse Index");
% yticks(64-flip(100:100:500))
% yticklabels(flip(100:100:500))

h.FontSize = 16;

