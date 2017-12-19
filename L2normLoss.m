function Y = L2normLoss(X, C, dzdy)

assert(numel(X) == numel(C));  %Asseration Failed
n = size(X,1) * size(X,2);
if nargin <= 2     
  diff=X(:)-C(:);
  idx=find(abs(diff)<0.1);
  X(idx)=C(idx);    
  Y = sum((exp(C(:)).*(X(:)-C(:))).^2) ;  %why exp()???????
else
  assert(numel(dzdy) == 1);    
  diff=X(:)-C(:);
  idx=find(abs(diff)<0.1);
  X(idx)=C(idx);    
  Y = reshape((dzdy / n) * 2*(exp(C(:)).*(X(:)-C(:))), size(X));  
end

end