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