function [f5,h5] = exp5_figures()
load("exp5.mat","Periodogram","MUSIC","OMP","cresfreq",...bit
    "cvswinfreq","swinfreq");

numFigs = 6;

% Signal parameters
dr = 3e8/2/300e6;
r = 128*dr;
r_label = 0:r/4096:r-r/4096;

f5 = gobjects(1,numFigs*2);
h5 = gobjects(1,numFigs*2);

f5(1) = figure(501);
clf(f5(1))
h5(1) = create_figure_exp5(r_label,Periodogram,true);
saveFigPng(h5(1),"exp5_periodogram")

f5(2) = figure(502);
clf(f5(2))
h5(2) = create_figure_exp5(r_label,MUSIC,false);
saveFigPng(h5(2),"exp5_music")

f5(3) = figure(503);
clf(f5(3))
h5(3) = create_figure_exp5(r_label,OMP,false);
saveFigPng(h5(3),"exp5_omp")

f5(4) = figure(504);
clf(f5(4))
h5(4) = create_figure_exp5(r_label,cresfreq,false);
saveFigPng(h5(4),"exp5_cresfreq")

f5(5) = figure(505);
clf(f5(5))
h5(5) = create_figure_exp5(r_label,swinfreq,false);
saveFigPng(h5(5),"exp5_swinfreq")

f5(6) = figure(506);
clf(f5(6))
h5(6) = create_figure_exp5(r_label,cvswinfreq,false);
saveFigPng(h5(6),"exp5_cvswinfreq")
end

function h = create_figure_exp5(r_label,A,isY)
A = flip(flip(A(1:2:end,:),1),2);

A = 20*log10(abs(A/max(A(:))));
A(A<-40) = -40;

x = r_label/2;
numPulses = size(A,1);
y = 1:numPulses;

h = handle(axes());
mesh(x,y,A,"FaceColor","interp","EdgeColor","none")

xlabel("Relative Range / m");
view(h,2)

% xlim([2.5,12.5])
ylim([y(1),y(end)])

if isY
    ylabel("Pulse Index");
    yticks(64-flip(100:100:500))
    yticklabels(flip(100:100:500))
end

if ~isY
    yticklabels([""])
end

h.FontSize = 16;
end