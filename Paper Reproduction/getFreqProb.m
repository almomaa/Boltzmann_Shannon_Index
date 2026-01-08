function [p, H] = getFreqProb(Label)

[~,~,ic] = unique(Label);
H = accumarray(ic,1);
p = H./sum(H);
end