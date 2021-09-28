---
title: Titanic - Machine Learning from Disaster
subtitle: 

# Summary for listings and search engines
summary: Kaggle Titanic - Machine Learning from Disaster

# Link this post with a project
projects: []

# Date published
date: "2021-09-28T00:00:00Z"

# Date updated
lastmod: "2021-09-28T00:00:00Z"

# Is this an unpublished draft?
draft: false

# Show this page in the Featured widget?
featured: false

# Featured image
# Place an image named `featured.jpg/png` in this page's folder and customize its options here.
image:
  caption: ''
  focal_point: ""
  placement: 2
  preview_only: false

authors:
- admin

tags:
- data mining

categories:
- blog
---

##  

{{< hl >}} _Switch to dark mode for better readability_ {{< /hl >}}

#### Hypothesis score : 0.77511

![png](./score1.png)

#### Final submission _improved_ score : 0.78708

![png](./score1.png)


[Scroll down to view the relevant code snippets, or click here to view the notebook on Kaggle](https://www.kaggle.com/saahilanande/saahildataminingprojecttutorial)

#### My Contribution : 

After the first submission of the default code provided by Kaggle I obtained a score of 0.775. To figure out how I could improve this performance I decided to look at other submissions in the competition and browse the web. While doing so I came across [Ken Jee's tutorial on youtube.](https://www.youtube.com/watch?v=I3FBJdiExcg&list=WL&index=1&ab_channel=KenJee) After going through the tutorial I realized I should explore the data and find out if there are any Null values. On doing so I found the category 'Age' had quite a few missing values. I then plotted the values of the category 'Age' on a histogram to see if the values are well-distributed or skewed to one side. On observing the histogram, I realised the values are fairly distributed and hence, I can use the mean of 'Age' to replace the missing values. I then decided to run the model on this updated data. But to my dismay I found that the new score was 0.77272, which was lower than the previous score.

In my next attempt to increase the score I thought of using a different model for prediction. After seeing fair results with the K Nearest Neighbors (KNN) model in [Ken Jee's Kaggle notebook ](https://www.kaggle.com/kenjee/titanic-project-example) I decided to implement it in my notebook. But this time again I noticed a further decrease in my score getting a score of 0.72727. So i attempted to adjust the parameters of KNN model, adjusting n-neighbors=3 which caused a further dip in my score fetching me a 0.70574.

Hoping to get better results, I tried using a Support Vector Machine algorithm (SVC in SKLearn), this time getting an even lower score of 0.66267!
In another attempt I tried to implement the Voting Classifier on the Random Forest Classifier, KNN, and SVM models. Doing this got me an increase in my score, getting me 0.77272 which was still lower than the first submission.

So, I went back to the data to explore a little more. I replaced the one missing null value in the 'Fare' attribute. Plotting a histogram of the fare attribute revealed that the data was skewed to one side and therefore needed to be normalised. I then created a new attribute named 'norm_fare' and added the normalised values of the fare to it. I added 'norm_fare' to the list of features to train the data. I chose Random Forest Classifier as the model this time since it fetched the best results. On running the model this time, I was finally able to increase my score to 0.78468. A small change in score but I did learn from it nevertheless.



![png](./1.png)

![png](./2.png)

![png](./3.png)

![png](./4.png)

![png](./5.png)

![png](./6.png)

![png](./7.png)

![png](./8.png)

![png](./9.png)

![png](./10.png)




