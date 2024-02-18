function [f4,h4] = exp4_figures()
load("exp4.mat","Periodogram","MUSIC","OMP","cresfreq",...
    "cvswinfreq","swinfreq")

numFigs = 6;

% Signal parameters
dr = 3e8/2/300e6;
r = 128*dr;
r_label = 0:r/4096:r-r/4096;

f4 = gobjects(1,numFigs*2);
h4 = gobjects(1,numFigs*2);

f4(1) = figure(401);
clf(f4(1))
h4(1) = create_figure_exp4(r_label,Periodogram,true,"big");
saveFigPng(h4(1),"exp4_periodogram_big")

f4(7) = figure(407);
clf(f4(7))
h4(7) = create_figure_exp4(r_label,Periodogram,true,"small");
saveFigPng(h4(7),"exp4_periodogram_small")

f4(2) = figure(402);
clf(f4(2))
h4(2) = create_figure_exp4(r_label,MUSIC,false,"big");
saveFigPng(h4(2),"exp4_music_big")

f4(8) = figure(408);
clf(f4(8))
h4(8) = create_figure_exp4(r_label,MUSIC,false,"small");
saveFigPng(h4(8),"exp4_music_small")

f4(3) = figure(403);
clf(f4(3))
h4(3) = create_figure_exp4(r_label,OMP,false,"big");
saveFigPng(h4(3),"exp4_omp_big")

f4(9) = figure(409);
clf(f4(9))
h4(9) = create_figure_exp4(r_label,OMP,false,"small");
saveFigPng(h4(9),"exp4_omp_small")

f4(4) = figure(404);
clf(f4(4))
h4(4) = create_figure_exp4(r_label,cresfreq,false,"big");
saveFigPng(h4(4),"exp4_cresfreq_big")

f4(10) = figure(410);
clf(f4(10))
h4(10) = create_figure_exp4(r_label,cresfreq,false,"small");
saveFigPng(h4(10),"exp4_cresfreq_small")

f4(5) = figure(405);
clf(f4(5))
h4(5) = create_figure_exp4(r_label,swinfreq,false,"big");
saveFigPng(h4(5),"exp4_swinfreq_big")

f4(11) = figure(411);
clf(f4(11))
h4(11) = create_figure_exp4(r_label,swinfreq,false,"small");
saveFigPng(h4(11),"exp4_swinfreq_small")

f4(6) = figure(406);
clf(f4(6))
h4(6) = create_figure_exp4(r_label,cvswinfreq,false,"big");
saveFigPng(h4(6),"exp4_cvswinfreq_big")

f4(12) = figure(412);
clf(f4(12))
h4(12) = create_figure_exp4(r_label,cvswinfreq,false,"small");
saveFigPng(h4(12),"exp4_cvswinfreq_small")
end

function h = create_figure_exp4(r_label,A,isY,formatType)
A = flip(flip(A,1),2);

x = r_label/2;
numPulses = size(A,1);
y = 1:numPulses;

h = handle(axes());
mesh(x,y,abs(A/max(A(:))),"FaceColor","interp","EdgeColor","none")

xlabel("Relative Range / m");
view(h,2)

if formatType == "big"
    formatBig(y,isY)
elseif formatType == "small"
    formatSmall(y,isY)
end

if ~isY
    yticklabels([""])
end

h.FontSize = 16;
end

function formatBig(y,isY)
xlim([2.5,12.5])
ylim([y(1),y(end)])
annotation("rectangle",[0.59,0.66,0.1,0.15],"Color","red","LineWidth",3)
if isY
    ylabel("Pulse Index");
    yticks(512-flip(100:100:500))
    yticklabels(flip(100:100:500))
end
end

function formatSmall(y,isY)
xlim([2.5,12.5])
ylim([y(1),y(end)])
if isY
    ylabel("Pulse Index");
    yticks(512-flip(100:20:500))
    yticklabels(flip(100:20:500))
end
xlim([8.4,9.7])
ylim([332,436])
xticks(0:0.5:20)
end