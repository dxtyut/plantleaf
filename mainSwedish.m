
% close all;
% clear;
clc;

%%
addpath(genpath('Liblinear'));
addpath(genpath('utilise'));
addpath(genpath('image'));
addpath(genpath('models'));

addpath(genpath(fullfile('matconvnet-1.0-beta25','matlab','mex')));
run(fullfile('matconvnet-1.0-beta25','matlab','vl_setupnn'));
% run(fullfile('matconvnet-1.0-beta25','matlab','vl_compilenn'));

%% Database Info
dataset_name = 'Swedish';    
fprintf('\n============== Experiments on %s ==============\n',dataset_name);
database = retr_database_dir(fullfile('.','image',dataset_name),'*.tif');

%% Networks
netnameset{1} = 'imagenet-vgg-verydeep-16'; 
netnameset{2} = 'imagenet-vgg-verydeep-19'; 
netnameset{3} = 'imagenet-resnet-50-dag';

convlayerset{1}=[9,16,23,30];
convlayerset{2}=[9,18,27,36];
convlayerset{3}=[90,100,110,120,130,140,152];

%% Extract Features
fprintf('============== Extract Features for all Images =====\n');
fprintf('============== Taking about 2.5 Hours ==============\n');
allfeat=zeros(11008,database.imnum);
for ii = 1:database.imnum        
    img = imread(database.path{ii});
    img = single(img) ; 
    feature = extFeaForOneImg(netnameset,convlayerset,img);
    allfeat(:,ii)=feature(:);
end

%%
nrounds=5;
ACC=zeros(1,nrounds);
for rr=1:nrounds
    %% Extract Features
    fprintf('\n\nRound %d\n============== Load Random Split Index ========================\n',rr);
    load([dataset_name,'_pre_random_matrix.mat'],'pre_rand_matrix_tr');
    tr_database = getSubBase(database,pre_rand_matrix_tr(rr,:));
    train_label = (tr_database.label)';
    load([dataset_name,'_pre_random_matrix.mat'],'pre_rand_matrix_ts');
    ts_database = getSubBase(database,pre_rand_matrix_ts(rr,:));
    test_label = (ts_database.label)';

    train_feat=allfeat(:,pre_rand_matrix_tr(rr,:));
    test_feat=allfeat(:,pre_rand_matrix_ts(rr,:));
    
    %%
    [train_feat_wpca,WPCA,meanimage_wpca] = dxWPCA(train_feat,min(size(train_feat)));
    test_feat_wpca = WPCA*bsxfun(@minus,test_feat,meanimage_wpca); 
    
    %% Perform SVM Classification
    fprintf('============== Performing SVM Training and Prediction =========\n');
    c = 10;    
    modellinearsvm = train(train_label', sparse(train_feat_wpca'), ['-s 1 -c ' num2str(c) ' -q']);
    [label_estimate, accuracy, decision_values] = predict(test_label(:),sparse(test_feat_wpca'), modellinearsvm, '-q');
    ACC(rr) = 100*sum(label_estimate(:)==test_label(:))/length(test_label);    
end

%% Results Report
fprintf('\n\n============== Results Report =================\n');
disp(ACC);
Ravg = mean(ACC);
Rstd = std(ACC);
fprintf('Average recognition accuracy: %f\n', Ravg);
fprintf('Standard deviation: %f\n', Rstd);    
fprintf('===============================================\n\n');



















