function [f2,h2] = exp2_figure()
load("exp2.mat","seperation","Periodogram","MUSIC","OMP",...
    "cresfreq","swinfreq","cvswinfreq")

separation = seperation';
Periodogram = Periodogram';
MUSIC = MUSIC';
OMP = OMP';
cResFreq = cresfreq';
SwinFreq = swinfreq';
CVSwinFreq = cvswinfreq';

f2 = figure(200);
clf(f2)
h2 = handle(axes());

LineWidth = 2;
LineStyle = "-";

plot(h2,separation,Periodogram,LineStyle,"LineWidth",LineWidth)
hold on
plot(h2,separation,MUSIC,LineStyle,"LineWidth",LineWidth)
plot(h2,separation,OMP,LineStyle,"LineWidth",LineWidth)
plot(h2,separation,cResFreq,LineStyle,"LineWidth",LineWidth)
plot(h2,separation,SwinFreq,LineStyle',"LineWidth",LineWidth)
plot(h2,separation,CVSwinFreq,LineStyle,"LineWidth",LineWidth)
hold off
h2.FontSize = 20;
fontname(h2,"Times New Roman")
grid on

legend("Periodogram","MUSIC","OMP",...,
    "cResFreq","SwinFreq","CVSwinFreq",...
    "FontSize",12)
xlabel("Frequency Separation Interval")
ylabel("Probability")
xlim([separation(1),1])

saveFigPng(h2,"exp2_res")
end