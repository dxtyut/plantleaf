function FEA=dxFeaRecali(inputtensor)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is our matlab implementation of the following paper:
% Kalantidis, Y.; Mellina, C.; Osindero, S. 
% Cross-Dimensional Weighting for Aggregated Deep Convolutional Features. 
% In Proceedings of the European Conference on Computer Vision Workshops, 
% Amsterdam, The Netherlands, 8–10 October 2016;  pp. 685–701.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    a=0.5;
    ainv=1/a;
    b=0.5;
    epsilong=1e-5;
    
    [h,w,c]=size(inputtensor);
    
    %% spatial
    S=sum(inputtensor,3);
    SSum=sum(sum(S.^a));
    S=(S/SSum^ainv).^b;
    S=repmat(S,[1,1,c]);
    locationfea=inputtensor.*S;
    locationfea=reshape(locationfea,[h*w,c]);
    locationfea=sum(locationfea,1);

    %% channel    
    inputtensor=reshape(inputtensor,[h*w,c]);
    inputtensor=inputtensor>0;
    omg=sum(inputtensor,1)/(h*w);
    omgall=sum(omg);
    
    channelfea=(c*epsilong+omgall)./(epsilong+omg);
    channelfea=log(channelfea);
    
    %%
    FEA=channelfea(:).*locationfea(:);
   