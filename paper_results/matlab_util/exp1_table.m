function [t_PSNR,t_SSIM] = exp1_table()
load("exp1.mat","snr","PSNR_cresfreq","PSNR_swinfreq",...
    "PSNR_cvswinfreq","SSIM_cresfreq","SSIM_swinfreq","SSIM_cvswinfreq")

snr = snr';

cResFreq = PSNR_cresfreq';
SwinFreq = PSNR_swinfreq';
CVSwinFreq = PSNR_cvswinfreq';

t_PSNR = table(snr,cResFreq,SwinFreq,CVSwinFreq);

cResFreq = SSIM_cresfreq';
SwinFreq = SSIM_swinfreq';
CVSwinFreq = SSIM_cvswinfreq';

t_SSIM = table(snr,cResFreq,SwinFreq,CVSwinFreq);

disp("PSNR")
disp(t_PSNR)

disp("SSIM")
disp(t_SSIM)
end