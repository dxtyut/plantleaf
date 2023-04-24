
function [xPCAWhite,W,Mean_Image] = dxWPCA(x,K)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  x:  D x N
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
[disc_set,disc_value,Mean_Image]  =  Eigenface_f(x,K);
u = disc_set;
s = diag(disc_value);

%%
% epsilon = 0.1;
epsilon = 1e-10;

W = diag(1./sqrt(diag(s(1:K,1:K))+epsilon))*u(:,1:K)';

xPCAWhite = diag(1./sqrt(diag(s(1:K,1:K))+epsilon))*u(:,1:K)'*bsxfun(@minus,x,Mean_Image);