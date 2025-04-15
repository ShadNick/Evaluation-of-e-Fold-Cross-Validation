import numpy
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

# Algorithm Regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

# Datasets Regression
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_diabetes

# Regressionsmodelle
regressions = [
    ("Decision Tree Regressor", DecisionTreeRegressor()),
    ("K-NNR", KNeighborsRegressor()),
    ("Lasso", Lasso()),
    ("Linear Regression", LinearRegression()),
    ("Ridge", Ridge())
]

# Datasets for Regression
datasets_regression = [
    ("Diabetes", load_diabetes()),
    ("Housing", fetch_california_housing())
]

k = 10 # Max number of splits

for model_name, model in regressions:
    print(f"Model: {model_name}")

    for dataset_name, dataset in datasets_regression:
        print(f"Dataset: {dataset_name}")
        method = dataset
        data = method.data
        target = method.target

        for run in range(1,101):
            counter = 0
            checker = False
            k_fold = KFold(n_splits=k, shuffle = True, random_state=run) # Splitting into k=10

            all_fold_scores = {}
            all_fold_meanscore = {}
            standard_deviation_after_fold = {}

            for fold_index, (train_index, test_index) in enumerate(k_fold.split(data,target), start = 1): # 1 to 10
                data_train = data[train_index]
                data_test = data[test_index]
                target_train = target[train_index]
                target_test = target[test_index]
                
                model.fit(data_train, target_train) # Training
                model_score = mean_absolute_error(target_test, model.predict(data_test)) # Evaluation

                all_fold_scores[fold_index] = model_score # Se = performance of model on fe

                mean_scores = numpy.mean(list(all_fold_scores.values())) # Me
                all_fold_meanscore[fold_index] = mean_scores 

                if fold_index > 1:
                    standard_deviation = numpy.std(list(all_fold_scores.values())) #standard deviation for samples
                    standard_deviation_after_fold[fold_index] = standard_deviation

                if fold_index > 2:
                    std = standard_deviation #deviation right now
                    last_standard_deviation = standard_deviation_after_fold[fold_index - 1] #previous

                    if std < last_standard_deviation:
                        counter += 1 # update stability counter
                    else:
                        absolute_change = abs(std - last_standard_deviation)
                        if absolute_change > 0.05 * last_standard_deviation:
                            counter = 0
                        else:
                            counter += 1

                if counter == 2:
                    if checker == False:
                        print(f"Stopped at k = {fold_index}")
                        print(f"Performance at {mean_scores}")

                        checker = True


            print(f"Run:{run}")
            print(f"Werte pro Durchlauf:")
            print(all_fold_scores)
            print(f"Durchschnittswerte pro Durchlauf:")
            print(all_fold_meanscore)
            print("Standardabweichung der Scores nach jedem Fold")
            print(standard_deviation_after_fold)







