function t_sep = exp2_table()
load("exp2.mat","seperation","Periodogram","MUSIC","OMP",...
    "cresfreq","swinfreq","cvswinfreq")

separation = seperation';
Periodogram = Periodogram';
MUSIC = MUSIC';
OMP = OMP';
cResFreq = cresfreq';
SwinFreq = swinfreq';
CVSwinFreq = cvswinfreq';

t_sep = table(separation,Periodogram,MUSIC,OMP,...
    cResFreq,SwinFreq,CVSwinFreq);

disp(t_sep)
end