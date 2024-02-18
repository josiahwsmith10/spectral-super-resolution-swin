function saveFigPng(h,fname)
exportgraphics(h,"saved_figures/"+fname+".png","Resolution",600)
saveas(h,"saved_figures/"+fname+".fig")
end