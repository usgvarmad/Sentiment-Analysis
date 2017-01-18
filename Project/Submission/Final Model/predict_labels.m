function [Y_hat] = predict_labels(word_counts, cnn_feat, prob_feat, color_feat, raw_imgs, raw_tweets)
load words_train.mat;
load colum_D_Bhargav.mat;
load NBconsprob.mat;

word_counts(word_counts>0) = 1;
[Y_hat, ys] = applyMNNB(0.5556, condprob, word_counts(:,cols_sel));                

end