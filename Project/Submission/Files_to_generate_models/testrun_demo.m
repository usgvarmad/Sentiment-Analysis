
%% Add path
addpath('./NB');
%% Naive Bayes
load('words_train.mat');
load('train_raw_img.mat')
load column_D_Bhargav.mat;
 
X = X(:,cols_sel);
[prior, condprob] = trainMNNB(X , Y);
[labels_NB, ys] = applyMNNB(prior, condprob, word_counts(:,cols_sel));     
 
%% Logistic Regression
load('words_train.mat');
load('train_raw_img.mat')
load column_D_Bhargav.mat;
 
Lambda = logspace(-6,-0.5,11);
load('words_train.mat');
load('train_raw_img.mat')
load column_D_Bhargav.mat;
 
X = X(:,cols_sel);
 Mdl_Logistic = fitclinear(X',Y,'ObservationsIn','columns',...
     'Learner','logistic','Solver','sparsa','Regularization','lasso',...
     'Lambda',Lambda,'GradientTolerance',1e-8);
 
MdlFinal_Logistic = selectModels(Mdl_Logistic ,7);
[labels_Logistic, ys_Logistic] =predict(MdlFinal_Logistic ,word_counts(:,cols_sel));

%% PCA with SVM on images
load('train_cnn_feat.mat');
 
[coeff_img]=pca((train_cnn_feat),'NumComponents',500);
SVMModel_img = fitcsvm(train_cnn_feat*coeff,full(Y),'KernelScale','auto','Standardize',true,'KernelFunction','RBF',...
    'OutlierFraction',0.1);
[labels_SVM, ys_SVM] =predict(SVMModel_img  ,cnn_feat*coeff_img);

%% LogitBoost 
load('words_train.mat');
load('train_raw_img.mat')
load column_D_Bhargav.mat;
 
 ensemble_logitBoost = fitensemble(full(X(:,cols_sel)),full(Y),'LogitBoost',1000,'Tree',...
    'type','classification');
[labels_ensemble,score_ensemble] = predict(ensemble_logitBoost ,full(word_counts(:,cols_sel)));
 
 
%% K-nearest Neighbors
load('words_train.mat');
 
 
mdl_knn = fitcknn(full(X(:,cols_sel)),full(Y),'NumNeighbors',3,'Standardize',1,'NSMethod','exhaustive');
[labels_KNN,score_KNN] = predict(mdl_knn  ,full(word_count));
 

%% SVM on words 
load('words_train.mat');
load('colum_D_Bhargav.mat');
SVMModel_words = fitcsvm(full(X(:,cols_sel)),full(Y),'KernelScale','auto','Standardize',true,'KernelFunction','linear',...
'OutlierFraction',0.1);
[labels_svm_words,score_svm_words] = predict(SVMModel_words,full(word_counts(:,cols_sel)));
 
 
%% Adaboost
load('words_train.mat');
load('train_raw_img.mat')
load column_D_Bhargav.mat;
 
 
 ensemble_adaBoost = fitensemble(full(X(:,cols_sel)),full(Y),'AdaBoostM1',1000,...
    'type','classification');
[labels_adaboost,score_boost] = predict(ensemble_adaBoost  ,full(word_counts(:,cols_sel)));
 
%% Cascading
 
load('train_cnn_feat.mat');
load('words_train.mat');
load('train_raw_img.mat')
load ('column_D_Bhargav.mat');
 
 
X = X(:,cols_sel);
[prior, condprob] = trainMNNB(X , Y);
[labels_NB, ys_NB] = applyMNNB(prior, condprob, word_counts(:,cols_sel));
 

[coeff_img]=pca((train_cnn_feat),'NumComponents',500);
SVMModel_img = fitcsvm(train_cnn_feat*coeff,full(Y),'KernelScale','auto','Standardize',true,'KernelFunction','RBF',...
    'OutlierFraction',0.1);
[labels_SVM, ys_SVM] =predict(SVMModel_img  ,cnn_feat*coeff_img);
 score_2 = ys_SVM';
      score_2 = bsxfun(@minus,score_2,mean(score_2));
     ys_SVM= score_2';
     new_Score = [ys_NB(:,1) ys_SVM(:,1)];
 
 
[labels_cascading] = predict(bag_new_score , new_Score );

%% PCA with LogitBoost

load('words_train.mat')
 coeff_words = pca(full(X),'NumComponents',1000);
 t = templateTree('Surrogate','on');
ensemble_logitboost_PCA = fitensemble(full(X*coeff_words),full(Y),'LogitBoost',1000,t,...
    'type','classification');
[labels_logistboost_PCA] = predict(ensemble_logitboost_PCA ,full(word_counts*coeff_words ));
 


