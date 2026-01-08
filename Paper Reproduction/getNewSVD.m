function x = getNewSVD(x,L,r)


for i=1:max(L)
    ix = (L==i);
    y = x(ix,:);

    [u,~,v] = svd(y,'econ');

    x(ix,:) = u*diag(r(i)*(ones(1,size(y,2))))*v';
end