import numpy
from sklearn.metrics import f1_score

# Algorithm Classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Algorithm Regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

# Datasets Classification
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.datasets import load_wine

# Datasets Regression
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_diabetes


# Modelle
classifications = [
    ("AdaBoost", AdaBoostClassifier(algorithm='SAMME')),
    ("Decision Tree Classifier", DecisionTreeClassifier()),
    ("Gaussian Naive Bayes", GaussianNB()),
    ("K-NN", KNeighborsClassifier()),
    ("Logistic Regression", LogisticRegression())
]

# Regressionsmodelle
regressions = [
    ("Decision Tree Regressor", DecisionTreeRegressor()),
    ("K-NNR", KNeighborsRegressor()),
    ("Lasso", Lasso()),
    ("Linear Regression", LinearRegression()),
    ("Ridge", Ridge())
]

# Datasets for Classifications
datasets_classifications = [
    ("Cancer", load_breast_cancer()),
    ("Iris", load_iris()),
    ("Digits", load_digits()),
    ("Wine", load_wine())
]

# Datasets for Regression
datasets_regression = [
    ("Housing", fetch_california_housing()),
    ("Diabetes", load_diabetes())
]

k = 10 # Max number of splits

for model_name, model in classifications:
    print(f"Model: {model_name}")
    all_runs_fold_scores = {}

    
    for run in range(1,101):
        counter = 0
        checker = False
        random_value = run # Different value for each run 
        k_fold = StratifiedKFold(n_splits=k, shuffle = True, random_state=random_value) # Splitting into k=10

        all_fold_scores = {}
        all_fold_meanscore = {}
        standard_deviation_after_fold = {}

        for dataset_name, dataset in datasets_classifications:
            print(f"Dataset: {dataset_name}")
            method = dataset
            data = method.data
            target = method.target

            if model_name == 'AdaBoost':
                model = AdaBoostClassifier(random_state=random_value)

            for fold_index, (train_index, test_index) in enumerate(k_fold.split(data,target), start = 1): # 1 to 10
                data_train = data[train_index]
                data_test = data[test_index]
                target_train = target[train_index]
                target_test = target[test_index]
                
                model.fit(data_train, target_train) # Training
                model_score = f1_score(target_test, model.predict(data_test), average='weighted') # Evaluation

                all_fold_scores[fold_index] = model_score # Se = performance of model on fe

                mean_scores = numpy.mean(list(all_fold_scores.values())) # Me
                all_fold_meanscore[fold_index] = mean_scores 

                if fold_index > 1:
                    standard_deviation = numpy.std(list(all_fold_scores.values()), ddof=1) #standard deviation for samples
                    standard_deviation_after_fold[fold_index] = standard_deviation

                if fold_index > 2:
                    std = standard_deviation #deviation right now
                    last_standard_deviation = standard_deviation_after_fold[fold_index - 1] #previous

                    if std < last_standard_deviation:
                        counter += 1 # update stability counter
                    else:
                        absolute_change = abs(std - last_standard_deviation)
                        absolute_percent = (absolute_change / std) * 100
                        if absolute_change > 5:
                            counter = 0
                        else:
                            counter += 1

                if counter == 2:
                    if checker == False:
                        print(f"Stopped at k = {fold_index}")
                        print(f"Performance at {mean_scores}")


                        checker = True

        all_runs_fold_scores[run] = all_fold_scores


        print(f"Run:{run}")
        print(f"Werte pro Durchlauf:")
        print(all_fold_scores)
        print(f"Durchschnittswerte pro Durchlauf:")
        print(all_fold_meanscore)
        print("Standardabweichung der Scores nach jedem Fold")
        print(standard_deviation_after_fold)







