function E = bsi(p, q)

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