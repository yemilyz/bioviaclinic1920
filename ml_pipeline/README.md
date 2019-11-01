# Introduction

For this assignment, you must work in teams of two or three. Your team can differ from last week. As a team, you can choose your own work style (from full pair programming to each tackling a separate component). Regardless of how you divide the work, every student is responsible for all work submitted by his or her group.

One member of your team should fork this repository and add the rest of the team as collaborators with write access. Each team member should then clone the forked repo to their machine.

Create a `partners.txt` file with the name of all team members. (Only one member needs to do this.)

Under `ml`, create a `results` folder to store your results for this week.

Your `HW3` directory should look as follows:

```bash
HW3
├── ml
|   ├── code
|   |   ├── classifiers.py
|   |   ├── datasets.py
|   |   ├── ml.py
|   |   ├── plot_results.py
|   |   └── preprocessors.py
|   ├── data
|   |   ├── iris.csv
|   |   └── titanic.csv
|   └── results
├── README.md
├── tictactoe
|   ├── oxo_cmd.py
|   ├── oxo_model.py
|   └── oxo_textui.py
└── writeup.md
```

You are allowed (and encouraged) to use any of your normal tools (e.g. File Explorer). But you should implement all functionality below in Python.

Please submit solutions to the written questions in *writeup.md*. **Include both your plots and your short responses.**



# Problem 1: Machine Learning Basics [7 pts; Time Estimate: 1 hr]

For this problem, you should work from the `HW3/ml` directory, e.g. `cd HW3/ml`. **You do not need to modify any code for this problem.**

We are going to black-box most of the machine learning this week and instead focus on *understanding the machine learning pipeline* and *interpreting results*. We chose this route because there is some overhead to making a ML pipeline reusable (in terms of allowing multiple datasets and classifiers).<sup>1</sup> And as we found out in-class, most data scientists spend relatively little of their time on the actual machine learning. But we hope that we at least whet your appetite and give you a taste of the process.

Some things to note:

- **datasets** : We have downloaded some datasets from [Kaggle](https://www.kaggle.com/), "the world's largest community of data scientists and machine learners". Under the `data` folder, you can find the datasets [iris](https://www.kaggle.com/uciml/iris) and [titanic](https://www.kaggle.com/c/titanic). **Note that the format of these datasets may differ from last week's assignments.** Please check out the respective Kaggle sites for more info.

- **classifiers** : For a baseline, we use `sklearn`'s [DummyClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html). We will also explore the classifiers discussed in class using `sklearn`'s [kNeighborsClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html), [RandomForestClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), and a [MLPClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html).

We are not providing as much help in the README or in the starter code this week. We want you to continue learning how you learn best. So please feel free to use any and all resources to help you understand the codebase. We highly recommend the [pandas](http://pandas.pydata.org/pandas-docs/stable/) and [scikit-learn](http://scikit-learn.org/stable/documentation.html) documentation! Remember that Google and Stack Overflow are your friends too.

<sup>1</sup> Our implementation uses some fancy CS concepts like code reflection that you may not have learned. (What is code reflection? According to Wikipedia, "the ability of a computer program to examine, introspect, and modify its own structure and behavior at runtime". It is pretty awesome but can also be dangerous, especially the "modify" part.) We also use some very high-level functions in `sklearn`; a more exhaustive treatment of ML (beyond the scope of a one-week assignment) would use more low-level functions.



## Code

The code is divided into several files: two scripts `ml` and `plot_results` and two modules `datasets` and `classifiers`. (You can probably guess the responsibility of each module.) There is an additional `preprocessors` module, but you will not need it for this problem.

### ml script
Run `python code/ml.py -h` to see the following help message:
```bash
usage: ml.py [-h] <dataset> <classifier>

positional arguments:
  <dataset>     [iris | titanic]
  <classifier>  [Dummy | KNN | RF | MLP]

optional arguments:
  -h, --help    show this help message and exit
```

As you might expect, you run `ml` by providing a dataset and classifier. So an example run might be the following:
```bash
python code/ml.py iris KNN
```

Let's walk through how the `ml` script works. The `main` function delegates all its responsibility to other functions. The "meat" of the code is in `run(datasest, classifier)`, which
1) Loads the chosen dataset from the `datasets` module, then splits the data into a training set and a test set.
2) Creates a `sklearn` [Pipeline](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) consisting of the chosen classifier from the `classifiers` module. A pipeline allows you to chain together a sequence of transforms followed by an estimator (e.g. a classifier).

    For now, you can think of a pipeline as just a classifier. Do not worry about the transforms; in Problem 2, we will add some preprocessors to this pipeline.
3) Tunes the pipeline using `sklearn`'s [RandomizedSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html). Once tuned using `fit`, running `predict` automatically applies the pipeline (the sequence of transforms and classifier with the best found parameters).

    Tuning a pipeline means simultaneously tuning all components of the pipeline. Again, for this problem, you can consider the pipeline as consisting of a single component, a classifier, so tuning means tuning the hyperparameters of the classifier.
4) Reports results on both the training set and test set.
5) Saves the pipeline to `<dataset>_<classifier>_pipeline.pkl` (a pickle file) and the results to `<dataset>_<classifier>_results.json` (a json file), both in the `results` directory.

### plot_results script
For a single dataset, once you have run `ml` using each classifier, you can make a pretty plot of the results. As before, run `python code/plot_results.py -h` to see the following help message:
```bash
usage: plot_results.py [-h] <dataset>

positional arguments:
  <dataset>   [iris | titanic]

optional arguments:
  -h, --help  show this help message and exit
```
The resulting plot is saved to `<dataset>_results.png` in the `results` directory.

So an example run might be the following:
```bash
python code/plot_results.py iris
```
(Remember, the above example command will not work unless you have first run `ml` using each classifier!)

## Try it out!

Run `ml` on the iris dataset using a dummy classifier. Then answer the following questions **in your own words** (do not copy-and-paste from other resources).

Questions 1 and 2 test your understanding of the pipeline and require that you browse some `sklearn` documentation, perhaps with the help of other resources. Questions 3 and 4 test your understanding of the results.

1) How does the dummy classifier make predictions? (1-2 sentences)

2) Briefly explain the method we used to determine the optimal values for hyperparameters. That is, in your own words, how does `RandomizedSearchCV` work? You can keep it simple for now and just assume the pipeline consists of a single classifier. (short paragraph, e.g. 5-10 sentences)

Now use kNN, RF, and MLP classifiers. (Watch out for capitalization. We use the standard "kNN" acronym in the readme, but the script expects `KNN`. Why? The script uses the argument to look up a class, and classes in Python start with a capital.) Once you have trained all the classifiers, make a plot using `plot_results`.

3) Why is the test performance higher than training performance? (1-2 sentences)

4) Using the plot, summarize your findings about the iris dataset. Do not overthink this -- we are not looking for anything that insightful here. Think of it this way: If you were to submit a quick report to a non-technical manager, what might you say about your investigations? (1-2 sentences)



# Problem 2: Machine Learning Advanced! [18 pts; Time Estimate: 3-4 hrs]

(Be sure to complete Problem 1 before tackling this one! Again, you should work from the `HW3/ml` directory. From this directory, we should be able to run `python code/<file>.py` and replicate all your work unless otherwise stated.)

Now that we have a pipeline in place, let us analyze the titanic dataset to determine what factors affect passenger survivability (morbid, yes, but a classic ML dataset).

**If you copy a substantial block of code (e.g. more than a few lines), please make sure you cite the resources (block or in-line comment).**

## a. Explore [5 pts; Time Estimate: 1 hr]

As with any new dataset, the first thing we should do is explore! In *writeup.md*, make some observations about the dataset. Your writeup should include at least four observations, backed by either descriptive statistics (e.g. mean, median, etc) or visualizations. Include at least one descriptive statistic and at least one visualization.

Your goal here is to gain some intuition for how features might affect survivability, so you should explore the data with this aim in mind. You do not have to explore *every* feature, but you should explore the ones that you think are most relevant. One way to approach this problem is to state some beliefs you have about survivability aboard the Titanic. Then ask yourself how you might justify these beliefs quantitatively or qualitatively.

Since this is exploratory, we will not worry too much about reusability. Go ahead and use whatever tool(s) you prefer. If you encounter an issue (e.g. missing data), make a reasonable choice and provide justification.

Do not worry about making your plots "pretty". Again, this is purely exploratory, so just make sure your plots are readable and support your observations.
    
If you are interested, for our visualizations, we made several histograms, one for each feature that we think affects survivability. Each plot has two overlapping histograms, with the color of the histogram indicating the class. Other visualizations are, of course, possible.

*Details*
- We like the combination of `pandas`, `seaborn`, and `matplotlib`. For a dataframe `df`, you can get summary statistics through `df.describe()`. Check out the documentation or tutorials for other useful features and visualizations.

    Titanic is a well-explored dataset, so feel free to use and modify any of the examples that you find online. Of course, make sure to include links to any resources you used in your comments.
- If you use Python, save your script as `explore_titanic.py` in the `code` folder. If you do not use Python, make sure to include in your writeup the tool that you did use and any supporting documents (e.g. an Excel file).
- So that you can include them in your writeup, save any visualizations in the `results` folder as `titanic_vis.png`. If you have multiple plots, use `titanic_vis0.png`, `titanic_vis1.png`, etc.

## b. Clean [2 pts; Time Estimate: 30 min]

The titanic dataset is more complicated and requires some data cleaning. Complete the missing portions of `titanic()` in the `datasets` module. All of the below requirements can be completed in 1-3 lines of code with `pandas` and `sklearn`. Remember what we said earlier -- use documentation and Google / StackOverflow to help out!

- We start by reading in a dataframe using `pandas` (already implemented).
- We expect that some features might be uninformative, so let us drop them. (An uninformative feature includes anything highly passenger-specific. Features with lots of unique values and with few samples per value are also problematic.) Drop the features `PassengerID`, `Name`, `Ticket`, and `Cabin`.
- The feature `Sex` is categorical and takes on values `female` and `male`. But many machine learning models (including kNN and MLP) require only numerical features. Map `female` to `0` and `male` to `1`.
- Similar, the feature `Embarked` is categorical and takes on values `C` (Cherbourg), `Q` (Queenstown), and `S` (Southampton). Since this feature is non-binary, we will use one-hot encoding instead. (You will want to find the `pandas` function that can convert a categorical variable to a new dataframe of indicator variables. Drop the `Embarked` feature that we will no longer use, and join the existing dataframe to the new one-hot encoded dataframe.)
- Drop any samples with a missing label.
- At this point, we have finished cleaning the dataset, so we extract the feature matrix `X` and label vector `y` (already implemented).
- Finally, the function can return the necessary variables (already implemented).

If you implemented everything correctly, then you should find the following summary statistics for `X` and `y`, obtained through `df.describe()`:
```bash
           Pclass         Sex         Age       SibSp       Parch        Fare           C           Q           S    Survived
count  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000  891.000000  891.000000  891.000000  891.000000
mean     2.308642    0.647587   29.699118    0.523008    0.381594   32.204208    0.188552    0.086420    0.722783    0.383838
std      0.836071    0.477990   14.526497    1.102743    0.806057   49.693429    0.391372    0.281141    0.447876    0.486592
min      1.000000    0.000000    0.420000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000
25%      2.000000    0.000000   20.125000    0.000000    0.000000    7.910400    0.000000    0.000000    0.000000    0.000000
50%      3.000000    1.000000   28.000000    0.000000    0.000000   14.454200    0.000000    0.000000    1.000000    0.000000
75%      3.000000    1.000000   38.000000    1.000000    0.000000   31.000000    0.000000    0.000000    1.000000    1.000000
max      3.000000    1.000000   80.000000    8.000000    6.000000  512.329200    1.000000    1.000000    1.000000    1.000000
```
  
## c. Preprocess [6 pts; Time Estimate: 1 hr]

Unfortunately, we are still not done! We need to add some preprocessors. In particular, the titanic dataset is missing values for some features, so we must infer these missing values through *imputation*. Also, unlike the iris dataset (in which all features are measurements in cm), the features for titanic are non-comparable, so we also need to do some feature *normalization* (by removing the mean and scaling to unit variance).

1. One approach is to impute and scale the dataset directly in `titanic()`, that is, after extracting the feature matrix and label vector but before returning these variables. Why is this a bad idea?

Our next goal is to complete the missing portions of tehe code so that we can correctly apply preprocessors.

2. Complete the missing portions in the `preprocessors` module. Specifically, using the `classifiers` module as a guide, implement two classes that inherit from the abstract base class `Preprocessor`.
    - One preprocessor should be named `Imputer`; transform using `sklearn`'s [Imputer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html), replacing missing values with the *most frequent* value for that feature; and have no hyperparameters.
    - Another preprocessor should be named `Scaler`; transform using `sklearn`'s [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html); and have no hyperparameters.

    *Miscellaneous*: Why do we have a `param_grid_` attribute if we never use it? To allow for future flexibility! Some preprocessors, for example, one that applies PCA to the data, would require hyperparameters.

    If you implemented everything correctly, then the `ml` script should be able to see the new preprocessors:
    ```bash
    usage: ml.py [-h] [-p <preprocessor>] <dataset> <classifier>

    positional arguments:
      <dataset>             [iris | titanic]
      <classifier>          [Dummy | KNN | RF | MLP]

    optional arguments:
      -h, --help            show this help message and exit
      -p <preprocessor>, --preprocessor <preprocessor>
                            [Imputer | Scaler]
    ```

3. Modify the specified sections of `ml.py` to use the preprocessors.
    - Start by modifying the function signatures and function calls to take in a list of preprocessors, e.g. `run(dataset, classifier)` becomes `run(dataset, preprocessor_list, classifier)`. Be sure to update the docstrings!
    - Update `make_pipeline(...)` to add the preprocessors as steps in the pipeline (before the classifier). Use the existing code as a guide.
    - In `run(...)`, modify `prefix` to include the preprocessors in the filename. The new filenames should be `<dataset>_<preprocessor>_<classifier>_pipeline.pkl` and `<dataset>_<preprocessor>_<classifier>_results.json`, where `<preprocessor>` is itself a `_`-separated list of preprocessors. That's complicated! Here is a concrete example: Applying an imputer, scaler, and kNN to the titanic dataset would generate the files `titanic_Imputer_Scaler_KNN_pipeline.pkl` and `titanic_Imputer_Scaler_KNN_results.json`.

4. Finally, apply your new pipeline! Here is an example:
    ```bash
    python code/ml.py -p Imputer -p Scaler titanic KNN
    ```
    If you implemented everything correctly, you should see a training accuracy of ~0.836 and a test accuracy of ~0.793.

    As before, apply all four classifiers, then make a plot of your results:
    ```bash
    python code/plot_results.py -p Imputer -p Scaler titanic
    ```

    Also try applying the classifiers with imputation but without scaling.

    Include both plots in your writeup, and summarize your findings. How do the classifiers perform relative to one another? How does scaling affect your results, and why do you think this is the case?

## d. Gain Insight [5 pts; Time Estimate: 1 hr]

If you look at `sklearn`'s `RandomForestClassifier`, you will see that it has the attribute `feature_importances_`.

1) In your own words, describe how you think feature importance might be calculated for this classifier. (You are free to look into how this attribute is actually calculated, but we are really looking for something simpler. Any justifiable hypothesis will do.) [2-3 sentences]

2) Write a small script `titanic_feature_importances.py` that loads the trained RF classifier (with imputation and scaling preprocessors), then either prints or plots the importance of the various features. If you plot, use whatever meaningful filename you would like, and save to the `results` folder.

    Include your results in your writeup. What do you observe? Is this surprising?

    *Hints*  
    - Check out `sklearn` documentation for recipes. A search with appropriately chosen key terms will save you loads of time!
    - You can load the trained RF from the pickle file using the following:
        ```python
        pipe = joblib.load(joblib_file)
        forest = pipe.named_steps['RF']
        ```

3) You might notice that neither `kNeighborsClassifier` nor `MLPClassifier` have a similar `feature_importances_` attribute. Why do you think this is the case? [2-3 sentences]

## Celebrate!
![Celebrate](https://media.makeameme.org/created/celebrate-good-times-7uhh3c.jpg)

Phew! We know that was a lot! On the plus side, now you have excellent starter code for future ML projects! (Clinic, wink wink.)



# Problem 3: UML [10 pts; Time Estimate: 2-3 hrs]

## Learning Goals
Your learning goals for this problem are multi-fold:
1) Practice reading and understanding an existing codebase! This is one of the most valuable skills that you will use post-college.
2) Practice using visual tools such as UML.
3) Practice working with the Model-View-Controller architecture.
For this problem, you should work from the `HW3/tictactoe` directory, e.g. `cd HW3/tictactoe`.

## The Problem
This week, rather than design your own tic-tac-toe game, we have provided an implementation for you. Run `python oxo_cmd.py`, and play around with the game. Try running a game straight through and also saving and resuming a game.

Next, open the associated files `oxo_cmd.py`, `oxo_model.py`, and `oxo_textui.py`.

The tictactode code is 437 lines including empty lines and comment blocks (and 185 lines of actual code), so a good starter size. We have purposefully not provided any external documentation.

## Your Tasks
To demonstrate your understanding:
1) Draw a UML class diagram. (You should have three classes.)
2) Draw UML sequence diagrams for the events that occur on a user's turn (from prompting the user to the next time the user is prompted). You can assume that the user selected a cell (rather than selecting to quit).
3) (Extra Credit) Identify at least one way in which you could improve the existing design! (1-2 sentences)

*Suggestions*  
- You do not have to understand *every* aspect of the code. Trying to pick out what events participate in the sequence diagram is part of the challenge.
- Different people have different strategies for working through code. Part of this problem is to help you figure out what works for you. Personally, I like to start from an entry-point (e.g. `main`) and trace the series of calls.
- Like last week, you may find it useful to draw the class and sequence diagram at the same time, using one to inform the other. Personally, when starting from code, I draw the class diagram first then use it to draw the sequence diagram.
- You may need fancier UML tools than the basics we discussed in-class, particularly for the sequence diagram. IBM has a nice [tutorial](https://www.ibm.com/developerworks/rational/library/3101.html) that shows off some more complex logic (e.g. conditionals, switches, loops, breaks).
- Do not spend more than two-to-three hours on this problem. We are not looking for the perfect diagram. As the learning goals state, we mostly want you to practice, practice, practice.

To submit, please scan any diagrams. Please make sure that anything hand-written or hand-drawn is readable! Then submit your diagrams in a single file `uml.pdf` with each diagram clearly labeled. (No need to include this one in your markdown writeup.)



# Submission

Please ensure that your `HW3` directory has the following file structure. Your directory may contain other files, but these are the ones we expect in your repo.
```bash
HW3
├── ml
|   ├── code
|   |   ├── classifiers.py
|   |   ├── datasets.py                      # with your modifications
|   |   ├── explore_titanic.py               # if you went this route for visualization
|   |   ├── ml.py
|   |   ├── plot_results.py
|   |   ├── preprocessors.py                 # with your modifications
|   |   ├── titanic_feature_importances.py   # created by you
|   |   └── titanic.xls                      # if you went this route for visualization
|   ├── data
|   |   ├── iris.csv
|   |   └── titanic.csv
|   └── results                              # your results
|       └── <lots of pkl, json, and png files here>
├── README.md
├── tictactoe
|   ├── oxo_cmd.py
|   ├── oxo_model.py
|   ├── oxo_textui.py
|   └── uml.pdf                              # your uml diagrams
└── writeup.md                               # with your modifications
```

Remember to update your fork with ALL changes you have made.  If you have used branches, merge your changes back into the master branch.

Because you have forked our repo, staff will have access to your fork on GitHub. Please do not submit a pull request; otherwise, other students will also have access to your repo. The state of your fork when the homework deadline hits will be considered your submission.