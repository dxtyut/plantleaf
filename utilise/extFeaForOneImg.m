
function feature = extFeaForOneImg(netnameset,convlayerset,I)

% tic;
feature = [];
for iii=1:length(netnameset)
    netname = netnameset{iii};
    layset = convlayerset{iii};
    
    if strcmp(netname,'imagenet-resnet-50-dag')==1
        net = dagnn.DagNN.loadobj(netname) ;
        
        net.mode = 'test' ;
        net.conserveMemory = 0;
    else
        net = vl_simplenn_tidy(load(netname));
    end
    averageColour_ = mean(mean(net.meta.normalization.averageImage,1),2) ;   
    imageSize_ = net.meta.normalization.imageSize;   
    
    img = preprocessImage(I, 1, imageSize_, averageColour_, 1, [0, 0]); 
    
    for jjj=1:numel(layset)      
        thislayer =layset(jjj);
        
        if strcmp(netname,'imagenet-resnet-50-dag')==1
            net.eval({'data', img}) ;
            
            %fm = net.vars(thislayer+1).value;     
            fm = net.vars(net.getVarIndex(net.vars(thislayer+1).name)).value;  
            
            %fm = net.vars(thislayer+1).value;    
            %fm1 = net.vars(net.getVarIndex(net.vars(thislayer+1).name)).value;       
            %assert(sum(sum(sum(fm-fm1)))==0,'error');
        else
            results = vl_simplenn(net, img) ;
            fm = results(thislayer+1).x;
        end
        
        tmp=dxFeaRecali(fm);clear fm;
        tmp=myNorm(tmp,'v');
        feature = [feature;tmp(:)];
    end
    clear net averageColour_ imageSize_;
end

feature=myNorm(feature,'v');

feature=double(feature);

% Rumtime=toc;
% fprintf('Time for extracting CNN features for one image is %f seconds\n',Rumtime);

























