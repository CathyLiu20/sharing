# DS Challenge
## Using NLP to predict upvotes based on headline
&nbsp;
### The Task
Attempting to make sense of human behaviors and actions is very interesting, and users' 'like' or 'upvote' for the news/articles may provide important insights for bussiness plans. My goal in this project is to predict the number of upvotes of some news/articles received based on their headline. Since upvotes are an indicator of popularity, I'd like to discover which types of news tend to be the most popular among users. 

### The Data
The data used for this projet consist of 509236 records of news title, numbers of upvotes and downvotes, author, news category, date and time created, and age restriction (whether over 18). No missing value appeared in the data. Below table shows the first five samples in the dataset. The feature 'up_votes' ranges from 0 to 21253 and is sparse at large values. The categoical features, 'down_votes' and 'category' only has one outcome so they are not very informative. 'Over_18' is binary but only 0.06% has a 'True' value, so it won't help much in the models. 'Author' contains 85838 unique values but hard to do the one-hot encoding. Thus, 'up_votes' and 'title' are of interests and used in the analysis.
![data](/data.png)


### NLP for headline

Next, the fun part is to analyze the title by NPL to extract the occurrence of words within them. The main procedures are word tokenization, removing stop words and stemming, and creating the bag of words and scoring the words. The Python word_tokenize function from nltk library is used to tokenization and TfidfVectorizer is used for scoring. 
The TFIDF frequency matrix is in the size of 509236 by 1728. 


### Predictive Models

Upvotes ('up_votes') is treated as a continuous variable and three models, linear regression, LASSO regression and XGBoost are built. The dataset is splited into train set (80%) and test set (20%). Since the hyperparameters tuning involving cross-validation, no validation set is separated from train set. The model performance is evaluated by mean squared error (MSE) with data samples in test set. 

Linear Regression 

Linear regression is straightforward and no hyperparameter tuning is involved. The MSE from test set is 273556.56 and RMSE is 523. The data is high-dimensional, containing 1728 features so that a feature selection technique (e.g., LASSO) may help the model generalize better to new data samples.

LASSO Regression 

LASSO provides a principled way to reduce the number of features in a model by adding a L1 peanlty factor. The 5-fold cross-validation is used to choose the 'best' penalty factor (alpha) based on MSE. The model is fitted again using the selected penalty factor. The MSE from test set is 273433.35, which is slightly lower than that from linear regression. 

XGBoost

XGBoost is an efficient implementation of gradient boosting algorithm, and it is computationally efficient and highly effective, especially for large data samples. Lots of hyperparameters can be tuned, and three important ones are selected due to computational cost. Specifically, learning rate, max_depth and min_child_weight are tuned with the 5-fold cross-validation. The MSE from test set is 277977.12, which is slightly larger than linear regression. A more exclusive hyperparameter tuning may likely to improve the model performance.  

### Summary 

The news headline is analyzed by NLP and three predictive models are built to use the words to predict user upvotes. LASSO regression outperforms the simple linear regression and XGBoost. The MSE are relatively large and this may due to the sparsity of the 'up_votes' feature. One may choose to categorize the upvotes and build classifers but need to be very careful about the 'cut-offs'. Another possible strategy is to select the top upvotes (e.g., top 10 percentiles) and their headline to study. It's likely that a stronger relationship can be found, and better models since smaller MSE achieved using current models but data samples only contain top 10 percent upvotes. Moreover, more computationaly efficient and powerful techniques can be used to process the data and build models.  
