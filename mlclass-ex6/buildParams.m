function Params = buildParams(C, sigma)
Params = zeros(length(C)*length(sigma),2);
for i=1:length(Params)
    Params(i,:) = [C(floor((i-1)/length(sigma))+1), sigma(mod(i-1,length(sigma))+1)];
end
end
