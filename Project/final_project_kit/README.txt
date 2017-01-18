The following is a description of the files:                                                                     

You will be given tweets and corresponding images and your job is to correctly classify them as joy/sad.

You will get:
Training:
4500 training examples with labels
4500 training examples without labels

We will have a leaderboard that displays your results and ranks on the testing set of 4500 examples.

For your final evaluation we will test your algorithm on a validation set of 4500 examples.


File Description:
1) color_description.csv
	- file with description of the 32 features for train_color and train_unlabeled_color.mat
2) reshape_img.m
	- MATLAB script that converts the 1x30000 rows in train_raw_img.mat and train_unlabeled_raw_img.mat to an image
3) topwords.csv
	- The dictionary of words in descending order of counts
4) predict_labels.m
	- MATLAB function template giving the format for code submitted to the leaderboard
	  We will call this function on our end to test your algorithms.

5) train_set/
The rows of each matrix correspond across all the features.

For example, you can get the correct label for the ith row of the
train_cnn_feat.mat feature by looking at the ith entry of the label
matrix in the words_train.mat file.

  - raw_tweets_train.mat
	- Contains a 4500x1 matrix indicating the tweet ids
	- Contains a 4500x1 struct containing the raw formats of the tweets
  - words_train.mat
	- Contains a 4500x10000 matrix indicating the tweet word counts
	- Contains a 4500x1 matrix indicating the tweet label (1 is joy, 0 is sad)
	- Contains a 4500x1 matrix indicating the tweet id
  - train_raw_img.mat
	- Contains a 4500x30000 matrix corresponding to the raw 100x100x3 image
  - train_cnn_feat.mat
	- Contains a 4500x4096 matrix that correspond to the last layer of a CNN
  - train_img_prob.mat
	- Contains a 4500x1365 matrix that corresponds to the scene/object probabilities
  - train_color.mat
	- Contains a 4500x32 matrix that corresponds to color features
  - train_tweet_id_img.mat
	- Contains a 4500x1 matrix that corresponds to the tweet id for the corresponding rows in the other feature matrices
  
6) train_unlabeled_set/
  All the same as in train_set except:
  - words_train_unlabeled.mat
	- Contains a 4500x1000 matrix indicating the tweet word counts
	- There is no label
		
	


