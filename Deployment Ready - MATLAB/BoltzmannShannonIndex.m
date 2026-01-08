function [BSI] = BoltzmannShannonIndex(Data, Label, varargin)

BSI = ClustersBSIE(Data,Label)

end


function E = ClustersBSIE(X,L)

[~,~,L] = unique(L,"rows");

p = getFreqProb(L);
q = getSVDProb(X,L);

E = distributions_bsi(p,q);
end

function [p, H] = getFreqProb(Label)

[~,~,ic] = unique(Label);
H = accumarray(ic,1);
p = H./sum(H);
end

function [q, V] = getSVDProb(Data,Label)



S = [];
for i=1:max(Label)
    ix = (Label==i);
    x  = Data(ix,:);
    [~,s,~] = svd(x-mean(x),'econ');
    S = cat(1, S, diag(s)');
end

W = cumsum(S.^2,2)./sum(S.^2,2);
B = (W>0.975);
for i=1:size(B,2)
    if all(B(:,i))
        S = S(:,1:i);
        break
    end
end
S
V = prod(S,2);
q = V/sum(V);
end

function E = distributions_bsi(p, q)

ix = or(~p , ~q);

if all(ix)
    E = 0;
    return;
end

p(ix) = []; 
q(ix) = [];

m = 0.5*(p+q);

E = 1-0.5*( dot(p,log2(p./m)) + dot(q,log2(q./m)));

end