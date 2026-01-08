function E = ClustersBSIE(X,L)

[~,~,L] = unique(L,"rows");

p = getFreqProb(L);
q = getSVDProb(X,L);

E = bsi(p,q);
end