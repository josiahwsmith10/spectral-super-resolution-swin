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

        if snr(indSNR) == 0
            mindB = -80;
        elseif snr(indSNR) == 20
            mindB = -120;
        end

        create_figure_exp3(h3(indSample,indSNR),Periodogram_i,MUSIC_i,...
            OMP_i,cResFreq_i,SwinFreq_i,CVSwinFreq_i,f_i,mindB);

        saveFigPng(h3(indSample,indSNR),"exp3_SNR"+snr(indSNR)+"dB_"+indSample)
    end
end
end

function create_figure_exp3(h,Periodogram,MUSIC,OMP,cResFreq,SwinFreq,...
    CVSwinFreq,f,mindB)
Periodogram = normalize_exp3(Periodogram);
MUSIC = normalize_exp3(MUSIC);
OMP = normalize_exp3(OMP,mindB);
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
    "Location","SW","FontSize",12)
h.FontSize = 20;
fontname(h,"Times New Roman")
grid on
xlabel("f / Hz")
ylabel("Normalized Power / dB")

xlim([round(min(f)-0.02,2),round(max(f)+0.02,2)])
ylim([mindB,0])
end

function y = normalize_exp3(x,lower_thr)
y = db(x/max(x(:)));
if nargin == 2
    y(y<lower_thr) = lower_thr-10;
end
end