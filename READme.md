# Classification of Subreddit Post Titles using Natural Language Processing and Supervised Machine Learning Models

#### Author: Holly Bok

## Problem Statement

The goal of this project is to create a supervised machine learning classification model that can correctly identify which Reddit.com subreddit a particular post title is from. I will gather post titles from the main pages of four subreddits and use the words and phrases in the titles to drive two classification models, a Logistic Regression and a Random Forest model. These models will be used to predict the origin subreddit given a post's title. The subreddits used in this project are: /r/Atlanta, /r/Austin, /r/Denver, and /r/SanFrancisco.

Post titles will be input into the models, predictions of subreddit origin will be made, and the strength of the models will be evaluated. All models will be scored using accuracy scores (the total number of correct predictions made by the model divided by the total number of predictions) with a range of possible values from 0 (no correct predictions) to 1 (all correct predictions). Accuracy scores for each model will be compared to one another and to the baseline score. The baseline score for this dataset is 0.25, which represents how accurate predictions would be if we simply guessed the majority class for every input. Each subreddit class has equal representation in this dataset, meaning guessing one class for all inputs would result in a .25 accuracy score. The goal of this project is to create a model that can predict subreddits of origin with a higher accuracy.

In addition to building a successful predictive model, the purpose of this project is to explore the similarities and differences between Atlanta, Austin, Denver, and San Francisco. These subreditts are chosen because they are of particular interest to me. Upon graduation from the General Assembly Data Science Immersive I am going to begin applying for positions in a variety of different cities. I have lived in Atlanta for almost 9 years, but I was born and raised in San Francisco and I am considering moving back home. I have narrowed down my job search to these four cities and I am very interested in what the quality of life is for the people who live there and the types of things that residents are discussing.

For my own enjoyment (and hopefully yours!) I have created a space in this repository to test specific words and phrases and see what city the model predicts for that phrase. This tool is found in the '02 - Modeling - Logistic Regression' sheet, at the very bottom, under the 'Phrase / Word Tester' subheader. For example, when input the fake post title

In this project I will create two supervised maching learning classification models that predict the subreddit of origin of a given post. The models will be compared for accuracy and the best model be used for data exploration. Results, trends, and patterns based off of the predictions from this model will be identified and conclusions will be drawn. Lastly, suggestions for further analysis will be discussed and problems within this model will be addressed.


## Gathering and Cleaning Data

Data for this project is collected from the website www.Reddit.com, an active message board in which users can interact on different subreddits related to specific topics and share links, news, thoughts, comedy, questions, and more. Subreddits represent communities of people who self identify as sharing an interest. This project explores patterns and trends in the type of posts made in four different subreddits from major cities in the United States: 1) Atlanta, GA, 2. Austin, TX, 3. Denver, CO, and 4. San Francisco, CA.

All data has been collected using the pushshift.io Reddit API, an API that was designed by Reddit users with the purpose of making Reddit's data more approachable for research and curiosity purposes. This project was made possible by the /r/datasets moderators, specifically /u/stuck_in_the_matrix who maintains the submissions archives where this data was gathered from. Data is gathered for submissions and is obtained using the base url 'https://api.pushshift.io/reddit/search/submission'.

The selected subreddits for each city were chosen because they are the largest and most subscribed to subreddits from each city of interest. There are other, smaller subreddits that are related to these cities, but only the largest and most popular from each were selected. Additionally, subreddits were chosen for exact location. For example, a subreddit for the 'Bay Area' exists that is larger and more popular than the 'San Francisco' subreddit, but /r/SanFrancisco is chosen because it focuses on the city of San Francisco specifically as opposed to the entire Bay Area. All four subreddits are considered to be active and well established (/r/SanFrancisco has 144,923 subscribers, /r/Austin has 153,532, /r/Atlanta has 123,493, and /r/Denver has 119,630). All subreddits have been in existence for 11 years, with the exception of /r/SanFrancisco which has been established for 12 years.

1000 posts are gathered from each subreddit and a DataFrame is created with 4,000 rows (each row representing a single post) and 2 columns (one for subreddit of origin and one for the content of the post title). This DataFrame is written to .csv as 'df.csv' and is saved in the 'data' folder of this repository for use in modeling.


## Modeling

The two classification models created are a Logistic Regression and a Random Forest. The accuracy of these models will be compared to the baseline and to each other, and predictions from the most accurate model will be used for data exploration.

The X variable, or independent variable, for both models is the collection of posts from all subreddits.
The y variable, or the dependent variable, is the subreddit that the post came from. The model will be predicting values for this variable using the X variable.

In both models, a train_test_split() is run on the X and y variables to split the data into two subsets: test and train. The purpose of this split is to use one subset, the training subset, to 'train' or teach our model how to make predictions. The model is then tested on the unseen testing data to assess how well the model can predict information it has not yet learned.

Post content is transformed for both the Logistic Regression and the Random Forest using the CountVectorizer. The CountVectorizer is a 'bag of words' transformer that takes in a set of posts and puts out an integer based vector that shows a count of which words were used in which subreddits. Put another way, the CountVectorizer transforms the posts into numerical data that can then be used in classification models to make predictions. The CountVectorizer is fit to the training data and is used in conjunction with both the Logistic Regression and the Random Forest. I have also instantiated the CountVectorizers with english stop words. Stop words are common words such as "are," "as," and "and" that are not included in the vector.


#### Logistic Regression

**Accuracy Scores:**

*train*: 0.988

*test:* 0.635

The logistic regression model is created using a Pipeline that transforms using the CountVectorizer and fits a Logistic Regression object to the training data. In order to create the most effective model with the highest accuracy score, the hyperparameters for the CountVectorizer are changed from the default. The best hyperparameters are chosen through a GridSearchCV in which different hyperparameters are tested and the model with the hyperparameters that make the most accurate predictions is chosen as the final Logistic Regression model. In this model, three hyperparameters are explored for the CountVectorizer. These features are: max_features (the total number of words or phrases recognized and saved), max_df (the percentage of the posts in the DataFrame that the word may be present in. Words / phrases that are in a higher percentage of posts are considered too common to aid in classification), and ngram_range (the size of anagrams / phrases that can be broken down and considered one individual).

The Logistic Regression model GridSearchCV returned a best model with no max features (the default), max_df of 0.8 (below the default), and an ngram_range of (1,1)(the default).

This model was scored on both the training and testing subsets.

#### Random Forest

**Accuracy Score:**

*train*: 0.868

*test:* 0.614


The Random Forest model is created using a similar Pipeline to the Logistic Regression, but utilizes the RandomForestClassification() function. The hyperparameters that are tested for this model are the hyperparameters for the Random Forest model itself rather than the CountVectorizer. The hyperparameters explored are: n_estimators (the number of decision trees that will be created in the model), max_depth (how deep each tree in the model goes), min_samples_split (the minimum count of a sample that will be required to create a split), and min_samples_leaf (the minimum count of a sample that are required to create an end leaf).

The Random Forest model GridSearchCV returned a best model with n_estimators of 100 (the default), no max_depth (the default), min_samples_split of 2 (the default), and min_samples_leaf of 2 (above the default).

The model was scored on both the training and testing subsets.

#### Logistic Regression vs. Random Forest

Overall, the Logistic Regression model performed better than the Random Forest model. As this model performed better, it is utilized in making predictions and identifying trends and patterns. Predictions for the subreddit of origin are made using the Logistic Regression model, input into a PANDAS DataFrame with the true subreddit of origin and the post titles, and written to .csv. Predictions are saved as 'titles_df.csv' in the 'data' folder for this repository. Additionally, a second .csv is written that contains the same structure but predicts subreddits for individual words as opposed to full titles. This is saved in the same folder as 'words_df.csv'. All data exploration hereafter in this READme, as well as sheet '04 - Exploration' of this repository, uses predictions generated using the Logistic Regression model.

It is worth noting that the test score for the Random Forest model (0.868) is lower than other scores achieved by the same model with a different set of hyperparameters. In other versions of this model the training accuracy was as high as 0.997, but in these cases the testing score was lower than in the model I selected. The selected hyperparameters resulted in the lowest score of all tested Random Forest combinations, but resulted in the highest accuracy score for the training subset.


## EDA

The model predicted a relatively equal total number of post titles for each city (1039 for Austin, 1005 for San Francisco, 980 for Atlanta, and 976 for Denver).

Out of 4,000 total predictions, the Logistic Regression model accurately predicted 3,585. Of the 415 incorrect predictions, 133 were incorrectly predicted to be from the Austin subreddit, 95 from Atlanta, 95 from Denver, and 92 from San Francisco.

DataFrames are made for all titles predicted for each city, all titles correctly predicted for each city, and all titles incorrectly predicted for each city. Additionally, DataFrames are made for the list of individual words attributed to each city. The number of total words attributed to each city had a very wide range; 3022 words are attributed to Austin, 1189 to Atlanta, 1075 to Denver, and 795 to San Francisco.

A SentimentIntensityAnalyzer is instantiated and all titles and words are given polarity scores. Polarity scores are given in 4 categories: neutral, positive, negative, and compound. Each word or phrase is given scores for each category for its sentiment. Scores for neutral, positive, and negative range from 0 to 1 and total 1.0. A word or title that is very positive, such as "delighted," would have a positive score close to, or at, 1 as well as negative and neutral scores close to, or at, 0; A word such as "kill" would have a negative score close to 1 and positive and neutral scores close to 0. The last score, the compound score, ranges from -1 to 1 and represents the overall sentiment of the input based on its positive, neutral, and negative scores. Very negative phrases score close to -1 and very positive phrases score close to 1. All titles and words are input to the SentimentIntensityAnalyzer and saved.

Several charts are created using matplotlib to compare the sentiment analysis of different cities. A chart is created to show how positive and negative the predicted post titles are for each city (Fig 1). Another of the same chart is created with the addition of the composite score (Fig 2). Additionally, a third chart is created to show the sentiment analysis of individual words by predicted city, as opposed to entire titles (Fig 3). Separate charts are created for only incorrectly identified posts. These charts show the composite scores for all misclassified posts by city (Fig 4). Lastly, charts are created to show the spread in sentiment for "passion words" and "passion titles" (Fig. 5). Passion words and titles refer to inputs that score above or below a 0 for sentiment. These words and phrases are particularly negative or positive as evaluated by the SentimentIntensityAnalyzer.

Fig. 1
![figure1](/figures/fig_1.png)

Fig. 2
![figure2](/figures/fig_2.png)

Fig. 3
![figure3](/figures/fig_3.png)

Fig. 4
![figure4](/figures/fig_4.png)

Fig. 5
![figure5](/figures/fig_5.png)

Individual title content can be seen and explored further in the sheet '04 - Exploration' of this repository under the subheader 'Content Exploration.' This section shows specific titles and words for each city and has further exploration of sentiment analysis.


## Conclusions

Overall, the Logistic Regression model preformed better than the Random Forest model. The Logistic Regression model had higher scores for both the training subset and the testing subset. The difference in the accuracy scores for the training subset (0.988 and 0.868) was larger than the difference in the testing subset (0.635 and 0.614). As the training scores are much higher than the testing scores, both models have evidence of overfitting. However, the Logistic Regression had the largest gap between the training and testing score, suggesting that the Logistic Regression model is more overfit to the training data than the Random Forest. In spite of overfitting, both models preform much, much better than the baseline score. The baseline accuracy score for this dataset is .25. This baseline score is still very low compared to the lowest score achieved by either model (0.614).

Although both models achieved lower accuracy scores for the testing subset (around 60-65%), this is still a good performance considering the nature of the classification. While these subreddits have some words that are certainly distinctive (such as names of local people or places) there are a lot of similar post types in all of the four subreddits. For example, it is probably just as likely to see a post titled "where is the best cheeseburger?" in San Francisco as it is in Denver. Since we are comparing subreddits of the same type (i.e. subreddits for major US cities) there will almost certainly be some overlap in post content that would be difficult for any model to differentiate. All of this considered, both models are preforming at a fairly decent rate.

The most interesting finding in terms of the accuracy of predictions is a bias towards Austin, TX. Austin had the largest number of total predictions (1039) and the largest number of incorrectly attributed posts (133), both by a wide margin. Additionally, the list of words attributed to Austin is huge compared to the three other cities (3022 as compared to 1189 for Atlanta, 1075 for Denver, and 795 for San Francisco). The majority of the posts that are incorrectly attributed to Austin were originally from the Denver subreddit (61 posts) followed by Atlanta (46) and San Francisco (26). The massive difference in words attributed to Austin as opposed to other cities is likely due to the fact that Austin had more unique words and/or longer post titles. I believe this is the case because there are more individual words attributed to Austin (suggesting the Austin post titles contained many unique words) but there are relatively equal number of post title predictions. This likely means that while Austin has a pure numbers bias the actual words that are used in Austin's post titles are not likely to be found on the subreddits for other cities. If the words were more likely to be seen in other cities I would expect the total number of posts incorrectly attributed to Austin to be larger. Put another way: Austin likely has a small lead on the others in terms of total predictions because there are more available words to aid in that prediction. However, these words must not be used that often because we don't see the same gap in predicted cities that we do in total number of words attributed to each city.

In comparison, the San Francisco subreddit had the smallest amount of total attributed words (795) while still maintaining a normal ratio of predicted posts (1005). This suggests that the words attributed to San Francisco are seen often and have high predictive power. This does not suggest that the words attributed to San Francisco are *stronger* or *better* words, but it does suggest the words attributed to San Francisco are words that we are going to see in other subreddits. Put another way, a word cannot be a good predictor if we don't even see the word in the inputs.

The sentiment analysis also provided insights into the differences between subreddit posts. In general, the vast majority of titles were ranked as a complete 0 for sentiment - meaning the post was entirely neutral. Of the titles that were considered positive or negative there was more variation in negative scores between predicted cities rather than the positive scores. ; The mean positive scores for each city are fairly consistent, with Denver slightly ahead and Austin slightly below. Negative scores had a larger variation, with Atlanta coming in as least negative and San Francisco as most negative. The composite scores for post titles follow this same trend, with Atlanta ranked as the most positive city and San Francisco as the least (with Austin coming in slightly above San Francisco and Denver below Atlanta). The sentiment analysis for individual words as opposed to post titles had different results. Denver came in last for both positive and negative scores while San Francisco was scored as the most positive.

Incorrectly predicted post titles were mostly positively ranked. This is likely due to the fact that highly positive titles are often ranked positive because they contain the word "best" in them when the post in title is actually asking for a recommendation for the best item (i.e. "where is the best burrito in town?"). These type of titles are difficult to distinguish and have positive rankings.

This speaks to a larger issue with the SentimentIntensityAnalyzer. While this analyzer is a very useful tool, it has a very difficult time understanding context of certain words. It is clear looking through the types of posts that are considered positive and negative that the analyzer works better for individual words than it does for posts. This problem is compounded by the fact sentiment scores for whole titles have a larger range than sentiment scores for individual words. Long phrases are thus more likely to be misranked or misunderstood as well as more likely to have a particularly positive or negative compound score.

In general, I was very surprised with my findings. I did not expect Atlanta to be the most positive city and I certainly did not expect San Francisco to be more negative. However, as the SentimentIntensityAnalyzer proved to be unreliable in some instances, I do not see it fit to conclude that any one city is particularly positive or negative based on this analysis.

The most interesting finding to me is the difference in the number of words that were attributed to San Francisco and Austin. Assuming that this difference *is* caused by Austin having longer and more word heavy, there could be a few implications. This could mean that Austin is the most active city. Perhaps there is so much going on and so much happening in Austin that people have a lot to say. On the other hand, it could mean that internet users in Austin are complaining or griping more. However, Austin did not rank particularly high for negativity or positivity so this is unlikely to be the case. Lastly, San Francisco having a shorter list of attributed words could be a sign that people in San Francisco are too busy or having too good of a time to make lengthy posts!



## Further Analysis

There are many ways this project could be continued or expanded. Firstly, I suggest the use of a different intensity or tone analyzer. The tone of the posts would be very interesting to explore further given a more reliable estimate of positivity and negativity. Additionally, I would like to test this model using a TfIdf instead of a CountVectorizer. It would be interesting to see if a different word transformer would produce any further insights.

I would also suggest including better filters and translators. Several of this cities, especially San Francisco and Austin, have many foreign language posts. Including a translator could help include these posts for a better complete picture of the posts coming from that city. Non-english stop words should also be included. Better filtering would also likely improve this model, such as removing advertisements and solicitations.

Lastly, I would like to see how this analysis changes when a larger number of posts are gathered and/or posts from different months or times of year. It would be interesting to see how cities reacted to major events such as national elections.
