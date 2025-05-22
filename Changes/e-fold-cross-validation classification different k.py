import numpy
import scipy.stats as stats
import matplotlib.pyplot as plt

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
    ("AdaBoost", AdaBoostClassifier()),
    ("Decision Tree Classifier", DecisionTreeClassifier()),
    ("Gaussian Naive Bayes", GaussianNB()),
    ("K-NN", make_pipeline(StandardScaler(), KNeighborsClassifier())),
    ("Logistic Regression", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)))
]

# Datasets for Classifications
datasets_classifications = [
    ("Cancer", load_breast_cancer()),
    ("Iris", load_iris()),
    ("Digits", load_digits()),
    ("Wine", load_wine())
]

def calc_confidence_interval(data, confidence_level = 0.95):

    se = numpy.std(data) / numpy.sqrt(len(data)) #Standarderror

    # Berechnungen 
    t_value = stats.t.ppf(1 - (1 - confidence_level) / 2, df=len(data)-1) # t-Wert for 95%
    lower_limit = numpy.mean(data) - t_value * se
    upper_limit = numpy.mean(data) + t_value * se

    return lower_limit, upper_limit

k = 5 # Max number of splits


for model_name, model in classifications:
    print(f"Model: {model_name}")
    
    all_stopped_check = {}
    all_confidence_check = {}

    for dataset_name, dataset in datasets_classifications:
        print(f"Dataset: {dataset_name}")
        stopped_at = {}
        data = dataset.data
        target = dataset.target
        difference = {}

        for run in range(1,101):
            counter = 0
            checker = False
            k_fold = StratifiedKFold(n_splits=k, shuffle = True, random_state=run) # Splitting into k=10

            all_fold_scores = {}
            all_fold_meanscore = {}
            standard_deviation_after_fold = {}
            stopped_fold = None

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
                        print(" ")
                        stopped_at[run] = fold_index
                        checker = True
                        stopped_fold = fold_index

            if stopped_fold is not None and stopped_fold != 5: # without not stopped and e=k=5 has same values
                difference_percent = abs(all_fold_meanscore[5] - all_fold_meanscore[stopped_fold]) / mean_score * 100
                difference[run] = difference_percent
            
            lower_limit, upper_limit = calc_confidence_interval(list(all_fold_scores.values()))

            if stopped_fold is not None:
                all_stopped_check[run] = stopped_fold

                if lower_limit <= all_fold_meanscore[stopped_fold] <= upper_limit:
                    all_confidence_check[run] = True
                else:
                    all_confidence_check[run] = False


        # Statistics
        print("----------Interval check----------")
        print(" ")

        print(f"Amount: {len(all_confidence_check)}")
        print(f"In: {list(all_confidence_check.values()).count(True)}")
        print(f"Out: {list(all_confidence_check.values()).count(False)}")

        print(" ")

        print("----------Stopped Distribution----------")
        print(" ")

        stopped_at_values = list(stopped_at.values())
        value_counts = {}
        for val in stopped_at_values:
            if val in value_counts:
                value_counts[val] += 1
            else:
                value_counts[val] = 1

        for value in sorted(value_counts):
            percent = round((value_counts[value] / len(stopped_at)) * 100, 2)
            print(f"Value {value}: {percent}%")

        average = sum(value for value in stopped_at_values if isinstance(value, (int, float))) / len(stopped_at)
        print(f"Durchschnitt: {round(average,2)}")

        print(" ")

        print("----------Percent Difference----------")
        print(" ")

        print(sum(difference.values()) / len(difference))

        print(" ")

        # Plotting Interval check
        figure, axes = plt.subplots(figsize=(6, 5))
        
        axes.set_title("Amount in Interval:", fontsize=15)
        axes.set_ylabel("Amount")

        in_count = list(all_confidence_check.values()).count(True)
        out_count = list(all_confidence_check.values()).count(False)

        axes.bar(['In', 'Out'], [in_count, out_count], color=['green','red'])

        for i, count in enumerate([in_count, out_count]):
            axes.text(i, count, str(count), ha='center', va='bottom', fontsize=12)

        plt.tight_layout()
        plt.show()
        plt.close()

        # Plotting cancel distribution
        figure, axes = plt.subplots(figsize=(10, 6))

        axes.set_title("Early cancellation on fold:", fontsize=15)
        axes.set_xlabel("Runs")
        axes.set_ylabel("Folds")

        folds_value = [stopped_at[run] for run in stopped_at.keys()]

        axes.bar(stopped_at.keys(), folds_value, color='blue')
        axes.axhline(numpy.mean(folds_value), color='orange', linewidth=2)

        plt.tight_layout()
        plt.show()
        plt.close()

        # Plotting difference procents to k=10
        figure, axes = plt.subplots(figsize=(10, 6))

        axes.set_title("Procent difference to compared k=10", fontsize=15)
        axes.set_xlabel("Runs")
        axes.set_ylabel("Value")

        axes.bar(difference.keys(), difference.values(), color='silver')
        axes.axhline(sum(difference.values()) / len(difference), color='orange', linewidth=2)

        plt.tight_layout()
        plt.show()
        plt.close()
