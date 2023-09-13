function [f3,h3] = exp3_figures()
load("exp3.mat","snr","f","Periodogram","MUSIC","OMP",...
    "cresfreq","swinfreq","cvswinfreq")

nSNR = length(snr);
nSamples = size(cresfreq,2);

f3 = gobjects(nSamples,nSNR);
h3 = gobjects(nSamples,nSNR);
for indSNR = 1:nSNR
    for indSample = 1:nSamples
        f3(indSample,indSNR) = figure(300 + double(snr(indSNR)) + indSample);
        clf(f3(indSample,indSNR))
        h3(indSample,indSNR) = handle(axes());
        
        Periodogram_i = Periodogram(:,indSample,indSNR);
        MUSIC_i = MUSIC(:,indSample,indSNR);
        OMP_i = OMP(:,indSample,indSNR);
        cResFreq_i = cresfreq(:,indSample,indSNR);
        SwinFreq_i = swinfreq(:,indSample,indSNR);
        CVSwinFreq_i = cvswinfreq(:,indSample,indSNR);
        f_i = squeeze(f(indSample,:,indSNR));
        f_i = f_i(f_i ~= -10);

        create_figure_exp3(h3(indSample,indSNR),Periodogram_i,MUSIC_i,...
            OMP_i,cResFreq_i,SwinFreq_i,CVSwinFreq_i,f_i);

        saveFigPng(h3(indSample,indSNR),"exp3_SNR"+snr(indSNR)+"dB_"+indSample)
    end
end
end

function create_figure_exp3(h,Periodogram,MUSIC,OMP,cResFreq,SwinFreq,...
    CVSwinFreq,f)
Periodogram = normalize_exp3(Periodogram);
MUSIC = normalize_exp3(MUSIC);
OMP = normalize_exp3(OMP);
cResFreq = normalize_exp3(cResFreq);
SwinFreq = normalize_exp3(SwinFreq);
CVSwinFreq = normalize_exp3(CVSwinFreq);

fAxis = linspace(-0.5,0.5,length(Periodogram));

LineWidth = 1;
LineStyle = "-";

plot(h,fAxis,Periodogram,LineStyle,"LineWidth",LineWidth)
hold on
plot(h,fAxis,MUSIC,LineStyle,"LineWidth",LineWidth)
plot(h,fAxis,OMP,LineStyle,"LineWidth",LineWidth)
plot(h,fAxis,cResFreq,LineStyle,"LineWidth",LineWidth)
plot(h,fAxis,SwinFreq,LineStyle',"LineWidth",LineWidth)
plot(h,fAxis,CVSwinFreq,LineStyle,"LineWidth",LineWidth)

for indF = 1:length(f)
    xline(f(indF),"--r","LineWidth",3)
end
yline(0,"k")

hold off

legend("Periodogram","MUSIC","OMP",...,
    "cResFreq","SwinFreq","CVSwinFreq",...
    "Location","SW")
h.FontSize = 16;
fontname(h,"Times New Roman")
grid on
xlabel("f / Hz")
ylabel("Normalized Power / dB")

xlim([round(min(f)-0.02,2),round(max(f)+0.02,2)])
ylim([-80,0])
end

function y = normalize_exp3(x,lower_thr)
if nargin < 2
    lower_thr = -100;
end
y = db(x/max(x(:)));
y(y<lower_thr) = lower_thr;
end