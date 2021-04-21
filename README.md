# DS Challenge
## Using NLP to predict upvotes based on headline

## The Task
Attempting to make sense of human behaviors and actions is very interesting, and users' 'like' or 'upvote' for the news/articles may provide important insights for bussiness plans. My goal in this project is to predict the number of upvotes of some news/articles receive based on their headline. Since upvotes are an indicator of popularity, I'd like to discover which types of news tend to be the most popular among users. This project is done in Python 3.6. 

## The Data
The data used for this projet consist of 509236 records of news title, numbers of upvotes and downvotes, author, news category, date and time created, and age restriction (whether over 18). No missing value in the data. The feature 'up_votes' ranges from 0 to 21253 and is sparse at large values. The categoical features, 'down_votes' and 'category' only has one outcome so they are not informative. 'Over_18' is binary but only 0.06% has value 'True', so it won't help much in the models. 'Author' contains 85838 unique values but hard to do the one-hot encoding. Thus, 'up_votes' and 'title' are of interests and used in the analysis.

## NLP for Titles

Next, the fun part is to analyze the headline by NPL to extract the occurrence of words within them. The main procedures are word tokenization, removing stop words and stemming, and creating the bag of words and scoring the words. The Python word_tokenize function from nltk library is used to tokenization and TfidfVectorizer is used for scoring. 
The TFIDF frequency matrix is in the size of 509236 by 1728. 


## Predictive Models

Upvotes ('up_votes') is treated as a continuous variable and three regression models: linear regression, LASSO regression and XGB are built. The dataset is splited into train set (80%) and test set (20%). Since the hyperparameters tuning involving cross-validation, no validation set is separated from train set. The model performance is evaluated by mean squared error (MSE) with test set. 

Linear Regression 

Linear regression is straightforward and no hyperparameter tuning is needed. The MSE from test set is 273556.56 and RMSE is 523. The data is high-dimensional, containing 1728 features so that a feature selection technique (e.g., LASSO) may help the model generalize better to new data samples.

LASSO Regression 

LASSO provides a principled way to reduce the number of features in a model by adding a L1 norm peanlty factor. The 5-fold cross-validation is used to choose the penalty factor (alpha) based on MSE. The model is fitted again using the selected 'best' penalty factor. The MSE from test set is 273433.35, which is slightly lower than that from linear regression. 

XGBoost

XGBoost is an efficient implementation of gradient boosting algorithm, and it is computationally efficient and highly effective, especially for large data samples. Lots of hyperparameters can be tunned, and three important ones are selected due to computation efficiency. Specifically, learning rate, max_depth and min_child_weight are tunned with the 5-fold cross-validation. The MSE from test set is 277977.12, which is slightly larger than linear regression. An exclusive hyperparameter tuning may likely improve the model performance.  

## Summary 

The news headline is analyzed by NLP and three predictive models are built to use the words to predict user upvotes. LASSO regression outperforms the simple linear regression and XGBoost. The MSE are relatively large and this may due to the sparsity of the 'up_votes' feature. One may choose to categorize the upvotes and build classifers but need to be careful about the 'cut-offs'. Moreover, more efficient and powerful techniques can be used to process the data and build models.  
