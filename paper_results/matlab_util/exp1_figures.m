function [f1,h1] = exp1_figures()
load("exp1.mat","snr","PSNR_cresfreq","PSNR_swinfreq",...
    "PSNR_cvswinfreq","SSIM_cresfreq","SSIM_swinfreq","SSIM_cvswinfreq")

snr = snr';

% Create PSNR Plot
cResFreq = PSNR_cresfreq';
SwinFreq = PSNR_swinfreq';
CVSwinFreq = PSNR_cvswinfreq';

f1 = figure(100);
clf(f1)
h1 = create_figure_exp1(snr,cResFreq,SwinFreq,CVSwinFreq);
ylabel("PSNR (dB)")
% daspect([1,1,1])

saveFigPng(h1,"exp1_psnr")

% Create SSIM Plot
cResFreq = SSIM_cresfreq';
SwinFreq = SSIM_swinfreq';
CVSwinFreq = SSIM_cvswinfreq';

f2 = figure(101);
clf(f2)
h2 = create_figure_exp1(snr,cResFreq,SwinFreq,CVSwinFreq);
ylabel("SSIM")
% daspect([50,1,1])

saveFigPng(h2,"exp1_ssim")

f1 = [f1,f2];
h1 = [h1,h2];
end

function h = create_figure_exp1(snr,cResFreq,SwinFreq,CVSwinFreq)
LineWidth = 2;
LineStyle = "-o";

h = handle(axes());
plot(h,snr,cResFreq,LineStyle,"LineWidth",LineWidth)
hold on
plot(h,snr,SwinFreq,LineStyle',"LineWidth",LineWidth)
plot(h,snr,CVSwinFreq,LineStyle,"LineWidth",LineWidth)
hold off
legend("cResFreq","SwinFreq","CVSwinFreq",...
    "Location","SE")
h.FontSize = 20;
fontname(h,"Times New Roman")
grid on
xlabel("SNR (dB)")
end