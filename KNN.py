import pandas as pd
import math
from sklearn.impute import SimpleImputer

def distance(test_point, train_point):
    return math.sqrt(sum((feature1 - feature2) ** 2 for feature1, feature2 in zip(test_point, train_point)))

def KNN(training_set, y_train, testing_set, k):
    results = []
    for i in range(len(testing_set)):
        distances = []
        test_point = testing_set.iloc[i].to_numpy()
        for j in range(len(training_set)):
            train_point = training_set.iloc[j].to_numpy()
            dist = distance(test_point, train_point)
            distances.append((dist, y_train.iloc[j]))
        distances.sort(key=lambda x: x[0])
        top_k = [item[1] for item in distances[:k]]
        countCkd = sum(1 for label in top_k if label == "ckd")
        results.append("ckd" if countCkd > (k - countCkd) else "notckd")
    return results

def calc_accuracy(y_true, y_pred):
    return sum(y_true == y_pred) / len(y_true)

def main():
    path = "Kidney_Disease_data_for_Classification_V2.csv"
    df = pd.read_csv(path)
    df = df.sample(frac=1).reset_index(drop=True)

    
    categorical_cols = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]
    numerical_cols = ['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc']
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    
    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    
    target = df["classification"]
    patient_ids = df["id"]
    df = df.drop(columns=["classification", "id"])
    
    train_idx = df.sample(frac=0.7).index
    training_set = df.loc[train_idx]
    testing_set = df.drop(train_idx)
    
    training_set = pd.get_dummies(training_set, columns=categorical_cols)
    testing_set = pd.get_dummies(testing_set, columns=categorical_cols)
    
    missing_cols = set(training_set.columns) - set(testing_set.columns)
    for col in missing_cols:
        testing_set[col] = 0
    testing_set = testing_set[training_set.columns]
    
    numerical_cols = training_set.select_dtypes(include=['number']).columns
    for col in numerical_cols:
        train_min = training_set[col].min()
        train_max = training_set[col].max()
        if train_max != train_min:
            training_set[col] = (training_set[col] - train_min) / (train_max - train_min)
            testing_set[col] = (testing_set[col] - train_min) / (train_max - train_min)
        else:
            training_set[col] = 0
            testing_set[col] = 0
    
    y_train = target.loc[train_idx]
    y_test = target.drop(train_idx)
    patient_ids_test = patient_ids.loc[testing_set.index]
    
    training_set = training_set.astype(float)
    testing_set  = testing_set.astype(float)
    
    results = KNN(training_set, y_train, testing_set, 5)
    
    for pid, pred in zip(patient_ids_test, results):
        print(f"Patient {pid}: {pred}")
    print("Accuracy: ", calc_accuracy(y_test, results))
    
if __name__ == "__main__":
    main()