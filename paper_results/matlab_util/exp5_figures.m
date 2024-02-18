function [f5,h5] = exp5_figures()
data = load("exp5.mat","Periodogram","MUSIC","OMP","cresfreq",...bit
    "cvswinfreq","swinfreq");

methods = ["Periodogram","MUSIC","OMP","cresfreq","swinfreq","cvswinfreq"];

f5 = gobjects(1,length(methods)*2);
h5 = gobjects(1,length(methods)*2);

for ind = 1:length(methods)
    f5(ind) = figure(500+ind);
    clf(f5(ind))
    if methods(ind) == "Periodogram" || methods(ind) == "swinfreq"
        h5(ind) = create_figure_exp5_type1(data.(methods(ind)),true);
    else
        h5(ind) = create_figure_exp5_type1(data.(methods(ind)),false);
    end
    saveFigPng(h5(ind),"exp5_"+methods(ind))

    f5(ind+6) = figure(500+ind+6);
    clf(f5(ind+6))
    if methods(ind) == "Periodogram"
        h5(ind+6) = create_figure_exp5_type2(data.(methods(ind)),true);
    else
        h5(ind+6) = create_figure_exp5_type2(data.(methods(ind)),false);
    end
    saveFigPng(h5(ind+6),"exp5_"+methods(ind)+"_hrrp")
end

end

function h = create_figure_exp5_type1(A,isY)
A = fftshift(A,2).';

A = A(750:3250,130:167);

y = linspace(-0.225,0.225,size(A,1));
x = linspace(-0.225,0.225,size(A,2));

A = A/max(A(:));
A(A<0) = 0;
A = 20*log10(A);

A(A<-35) = -35;

h = handle(axes());
mesh(h,x,y,A,"FaceColor","interp","EdgeColor","none")

xlabel(h,"Cross Range / m");
view(h,2)

xlim(h,[x(1),x(end)])
ylim(h,[y(1),y(end)])

if isY
    ylabel(h,"Range / m");
end

if ~isY
    yticklabels(h,[""])
end

h.FontSize = 20;
end

function h = create_figure_exp5_type2(A,isY)
A = fftshift(A,2).';

A = A(750:3250,130:167);

y = linspace(-0.225,0.225,size(A,1));

A = A/max(A(:));
A(A<0) = 0;
a = A(:,end/2);
a = a/max(a(:));

h = handle(axes());
plot(h,y,a)

xlabel(h,"Range / m");

xlim(h,[0,1])
xlim(h,[y(1),y(end)])

if isY
    ylabel(h,"Normalized Amplitude");
end

if ~isY
    yticklabels(h,[""])
end

h.FontSize = 24;
end