# cis520_twitter_sentiment_analysis


% proj_final README.TXT

Group: GULTI
Members: 
Venkata Bharath Reddy Karnati (vbharath),
Uzval Gopinath Dontiboyina (uzval),
Bhargav Kalluri (bhargavk)


The four methods we have implemented are: 
Generative method: Naive Bayes
Discriminative methods:Logistic Regression, LogitBoost, SVM, Cascading, Adaboost
Instance-based method: K-nearest Neighbors  
Semi-supervised dimensionality reduction of the data :PCA with SVM on images 
   

****IMPORTANT******

a)We have included the generated models in the "Models" folder.
b)The final model that we got the highest accuracy for is there in the "Final Model" folder which can be executed by running the "predict_labels.m" file.
c)To run the code to generate these models go to "Files_to_generate_models" folder, go through "testrun_demo.m" file and load all the parameters that you would need to run "predict_labels.m".
(Note: "testrun_demo.m" file is a "predict_labels.m" file for all the methods)
d)"colum_D_Bhargav.mat" contains a mixture of features which have been selected post binormal separation and features which have the highest IG values. 
e)To test the models in the "Models" folder, Load them into workspace and run the "To test" code given in the method description.  
*******************



**********************
Method's Description:
***********************
1) Naive Bayes 
***********************
The Naive Bayes(Multinomial Naive Bayes) model was implemented for text (words) classification.
This model was implemented based on ‘Text Classification using Naive Bayes’ by Hiroshi Shimodaira and various online tutorials. 
 

To test: 
[labels_NB, ys] = applyMNNB(prior, condprob, word_counts(:,cols_sel));
*This was the model submitted as our final model.

Accuracy: ~80.64%

**************************
2) Logistic Regression 
**************************
We used "fitclinear" to perform Logistic Regression on the train data to generate our model.


To test: 
[labels_Logistic, ys_Logistic] =predict(MdlFinal_Logistic ,word_counts(:,cols_sel));

Accuracy: ~73.88%

************************
3) LogitBoost 
************************
We used "fitensemble" with "LogitBoost" and "1000" Trees as parameters on the train data to generate our model.

To test: 
[labels_ensemble,score_ensemble] = predict(ensemble_logitBoost ,full(word_counts(:,cols_sel)));

Accuracy: ~80.02%

**************************
4) SVM with PCA on images 
**************************
We used "fitcsvm" with RBF kernel to perform SVM on PCA'ed image train data to generate our model.

To Test: 
[labels_SVM, ys_SVM] =predict(SVMModel_img  ,cnn_feat*coeff_img);

Accuracy: ~ 64.63%

**************************
5) K-Nearest Neighbors
**************************
We used "fitcknn" with 3 nearest neighbors to generate our model.

To test: 
[labels_KNN,score_KNN] = predict(mdl_knn  ,full(word_count));

Accuracy: ~ 65.76%

*******************
6) SVM on words
*******************
We used "fitcsvm" with linear kernel to perform SVM on word train data to generate our model.

To test: 
[labels_svm_words,score_svm_words] = predict(SVMModel_words   ,full(word_counts(:,cols_sel)));


Accuracy: ~74.33%

*******************
7) Adaboost 
*******************
We used "fitensemble" with "AdaBoostM1" and "1000" Trees as parameters on the train data to generate our model.

To test:
    [labels_adaboost,score_boost] = predict(ensemble_adaBoost  ,full(word_counts(:,cols_sel)));
	

Accuracy: ~75.02%

*************
8) Cascading
**************
We performed "Naive Bayes" on words and "PCA with SVM" on images. Then finally performed "Logistic Regression" on normalized scores of above models. 


To test:
[labels_NB, ys_NB] = applyMNNB(prior, condprob, word_counts(:,cols_sel));
[labels_SVM, ys_SVM] =predict(SVMModel_img  ,cnn_feat*coeff_img);
 score_2 = ys_SVM';
      score_2 = bsxfun(@minus,score_2,mean(score_2));
     ys_SVM= score_2';
     new_Score = [ys_NB(:,1) ys_SVM(:,1)];
[labels_cascading] = predict(bag_new_score , new_Score );


Accuracy: ~80.24%

***************************
9)PCA with LogitBoost
****************************
We used "fitensemble" with "LogitBoost" and "1000" Trees as parameters on the PCA'ed train data to generate our model.

To Test:

[labels_logistboost_PCA] = predict(ensemble_logitboost_PCA ,full(word_counts*coeff_words ));


Accuracy: ~75.20%
