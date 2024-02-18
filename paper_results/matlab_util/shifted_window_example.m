function [fswin,hswin] = shifted_window_example()
%%
rng(1)

nfreqs = 3;
N = 128;

f_ell = rand(nfreqs,1) - 0.5;
a_ell = randn(nfreqs,1) + 1j*randn(nfreqs,1);
w = randn(1,N) + 1j*randn(1,N);

n = 1:N;

x = sum(a_ell.*exp(1j*2*pi*f_ell.*n),1) + w;

fswin = gobjects(1,5);
hswin = gobjects(1,5);

xMinVec = [1:64:65,-31:64:117];
xMaxVec = xMinVec+63;

yMin = min([min(real(x)),min(imag(x))]);
yMax = max([max(real(x)),max(imag(x))]);

for ind = 1:5
    fswin(ind) = figure(600+ind);
    hswin(ind) = create_figure_swin_example(fswin(ind),x,n,N,16,xMinVec(ind),xMaxVec(ind));
    ylim([yMin,yMax])

    saveFigPng(hswin(ind),"shifted_window_example_"+ind)
end

end

function h = create_figure_swin_example(f,x,n,N,spacing,xMin,xMax)

clf(f)
h = handle(axes());

plot(n,real(x),"b","LineWidth",3)
hold on
plot(n,imag(x),"r","LineWidth",3)

xlim([xMin-1,xMax])
yticks([])
xticks(-2*N:spacing:2*N)
grid on

daspect([4,1,1])

h.GridAlpha = 0.75;
h.FontSize = 16;
end