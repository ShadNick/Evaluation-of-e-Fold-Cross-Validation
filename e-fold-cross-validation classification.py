from collections import Counter
from itertools import count

import numpy
import scipy.stats as stats
from pandas import value_counts
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Algorithm Classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Datasets Classification
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.datasets import load_wine

# Modelle
classifications = [
    #("AdaBoost", AdaBoostClassifier()),
    #("Decision Tree Classifier", DecisionTreeClassifier()),
    #("Gaussian Naive Bayes", GaussianNB()),
    ("K-NN", make_pipeline(StandardScaler(), KNeighborsClassifier())),
    #("Logistic Regression", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)))
]

# Datasets for Classifications
datasets_classifications = [
    ("Cancer", load_breast_cancer()),
    #("Iris", load_iris()),
    #("Digits", load_digits()),
    #("Wine", load_wine())
]

def calc_confidence_interval(data, confidence_level = 0.95):

    data_length = len(data) 
    mean = numpy.mean(data) # Mittelwert
    se = numpy.std(data) / numpy.sqrt(data_length) #Standardfehler

    # Berechnungen 
    t_value = stats.t.ppf(1 - (1 - confidence_level) / 2, df=data_length-1) # t-Wert für 95%
    lower_limit = mean - t_value * se
    upper_limit = mean + t_value * se

    return lower_limit, upper_limit

k = 10 # Max number of splits


for model_name, model in classifications:
    print(f"Model: {model_name}")
    
    all_stopped_check = {}
    all_confidence_check = {}

    for dataset_name, dataset in datasets_classifications:
        print(f"Dataset: {dataset_name}")
        stopped_at = {}
        data = dataset.data
        target = dataset.target
        difference = []

        for run in range(1,101):
            counter = 0
            checker = False
            k_fold = StratifiedKFold(n_splits=k, shuffle = True, random_state=run) # Splitting into k=10

            all_fold_scores = {}
            all_fold_meanscore = {}
            standard_deviation_after_fold = {}

            for fold_index, (train_index, test_index) in enumerate(k_fold.split(data,target), start = 1): # 1 to 10
                data_train = data[train_index]
                data_test = data[test_index]
                target_train = target[train_index]
                target_test = target[test_index]
                
                model.fit(data_train, target_train) # Training
                model_score = f1_score(target_test, model.predict(data_test), average='weighted') # Evaluation

                all_fold_scores[fold_index] = model_score # Se = performance of model on fe

                mean_score = numpy.mean(list(all_fold_scores.values())) # Me
                all_fold_meanscore[fold_index] = mean_score 

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
                        print(f"Run:{run}")
                        print(f"Stopped at k = {fold_index}")
                        print(f"Performance at {mean_score}")
                        stopped_at[run] = fold_index

                        checker = True
                        stopped_fold = fold_index
                     
                if checker == False:
                    stopped_fold = fold_index #not stopped

                    
            difference_percent = abs(all_fold_meanscore[10] - all_fold_meanscore[stopped_fold]) / mean_score * 100
            difference.append(f"{difference_percent:.2f}%")
                        

            scores = numpy.array(list(all_fold_scores.values()))
            
            lower_limit, upper_limit = calc_confidence_interval(scores)

            all_stopped_check[run] = stopped_fold
            filtered_cancel_check = [value for value in all_stopped_check.values() if value == 10]
            if lower_limit <= all_fold_meanscore[stopped_fold] <= upper_limit:
                all_confidence_check[run] = True
            else:
                all_confidence_check[run] = False

            filtered_all_confidence_check = [value for key, value in all_confidence_check.items() if key not in filtered_cancel_check]


        print("------------------Interval check----------------------")     
        print(all_stopped_check)
        print(filtered_all_confidence_check)
        count_true = filtered_all_confidence_check.count(True)
        count_false = filtered_all_confidence_check.count(False)

        print("Interval drinne")
        print(len(filtered_all_confidence_check))
        print(count_true)
        print(count_false)


        print("---------Stopped Distribution---------------------------------")
        # Statistics
        value_counts = Counter(stopped_at.values())
        total = len(stopped_at)
        perecentages = {k: round((v / total) * 100, 2) for k,v in value_counts.items()}

        for value, percent in sorted(perecentages.items()):
            print(f"Wert {value}: {percent}%")

        values = list(stopped_at.values())
        average = sum(values) / len(values)
        print(f"Durchschnitt: {round(average,2)}")


        print("-----------Percent Differences----------------")

        print(difference)

        number_percent = [float(percent.strip('%')) for percent in difference]
        average_number = sum(number_percent) / len(number_percent)

        print(average_number)

        input("Button to Resume")
