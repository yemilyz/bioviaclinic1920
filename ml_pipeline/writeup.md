Place your results and responses to written questions in this file.

# Problem 1: Machine Learning Basics

## Try it out!

1. The dummy classifier works by making predictions
based on the distribution of the samples in the training
data. It therefore applies this same distrubution of samples
to the test data and makes the guesses according to that.

2. This function goes through the parameters involved, and 
runs the classifier for different parameter values. It runs
the processes specificed in the first pipeline parameter for the
function. The second paramter represents all of the parameters of
the classifiers and the distributions for the possible test parameters
which can be used. It then chooses different parameter values for these
parameters randomly a set number of times, which is the third paramter
into the function. 

3. It's possible that the test data set was extremely small, and so
all the labels were applied properly. It's also possible that the test
data was homoegenous already and therefore there wasn't any actual
identifying that had to be done.

4. Ultimately, it seems like there are very distinct features between 
the different iris types. They follow expected patterns and therefore
are easily discernable.

# Problem 2: Machine Learning Advanced!

## a. Explore

Using excel, we found that the survival rate was 38%.
We also checked the ages of people, in case that had any impact on their fitness and ability
to get off the boat. We saw that most people were below 35, and the ages really skewed towards the lower end. A huge majority of all of the people were between the ages of 17 and 35. Finally, we checked how many relatives were
on board for each person. We found that on average, each person had .90 relatives onboard. This suggests that while
many values were 0, most people had some relative on the ship which may have impacted when and if they decided to leave the ship. Finally, we checked if there was any interestign data in how much people paid for their tickets, because if people were wealthier that may have caused them to be saved first. I found that the median was 14 with a standard deviation of 49. This suggests that most people were not well off on this boat, and there were a couple of wealthy people who spent a lot of money on the ship.

## c. Preprocess

1. If we scale the data only after we take the feature matrix and the label vector, then we've already finished using some of the data, which may change. This can cause the usage of stale data.

4. Without scaling, there's a lot more variability in the correctness of the models with regards to the different metrics. While the average of the general metrics for all of the non dummy models seems to still be the same, there's a lot more variance. This is possible because these different models may take into effect the numbers and
features differently and they might have different weightages per each model, with different impacts on the models, because they're no longer scaled.

## d. Gain insight!

1. It could be based on the feature that has the most variance. This means that it is likely the biggest indicator for any different labels. It could also be based on the one that varies the least between each cluster, and therefore this gives more indication into how the things relate to each other.

2. The results are: [0.13106076 0.50429657 0.113146   0.02947395 0.02842193 0.17112666
 0.01074609 0.01172804]. What is interesting about this is that just one feature seems to have a large impact
 on the way the items are classified, which seems to be the gender. It's not too surprising since we also thought it would have an impact, but what is interesting is how high the weight on gender is in determining if people survived or not.

3. It's because neither of these use feature importance. Instead, they use the data from the neighbors around them and don't use the importance of features in determining what the classification of the data was.
