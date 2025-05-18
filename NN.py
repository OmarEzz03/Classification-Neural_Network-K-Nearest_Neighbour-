import pandas as pd
import math
import numpy as np
from sklearn.impute import SimpleImputer

def calc_accuracy(y_true, y_pred):
    return sum(y_true == y_pred) / len(y_true)

class OutputNode:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias
        self.inputs = None
        self.output = None
        self.error = None
    
    def calc_output(self, hidden_outputs):
        self.inputs = hidden_outputs
        z = np.dot(hidden_outputs, self.weight) + self.bias
        self.output = 1 / (1 + np.exp(-z)) 
        return self.output
    
    def calc_error(self, true_value):
        # self.error = (true_value - self.output) * self.output * (1 - self.output)
        self.error = self.output - true_value
        return self.error
    
    def update_weights(self, learning_rate):
        self.weight -= learning_rate * self.error * self.inputs
        self.bias -= learning_rate * self.error

class HiddenNode:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias
        self.inputs = None
        self.output = None
        self.error = None
    
    def calc_output(self, input_vector):
        self.inputs = input_vector
        z = np.dot(input_vector, self.weight) + self.bias
        self.output = 1 / (1 + np.exp(-z))
        return self.output
    
    def calc_error(self, downstream_weight, downstream_error):
        self.error = self.output * (1 - self.output) * downstream_weight * downstream_error
        return self.error
    
    def update_weights(self, learning_rate):
        self.weight -= learning_rate * self.error * self.inputs
        self.bias -= learning_rate * self.error

def main():
    path = "Kidney_Disease_data_for_Classification_V2.csv"
    df = pd.read_csv(path)
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle

    #region Preprocessing
    categorical_cols = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]
    numerical_cols = ['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc']
    
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    
    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    
    target = df["classification"].map({'ckd': 1, 'notckd': 0})
    patient_ids = df["id"]
    df = df.drop(columns=["classification", "id"])
    
    train_idx = df.sample(frac=0.7).index
    training_set = df.loc[train_idx]
    testing_set = df.drop(train_idx)
    
    training_set = pd.get_dummies(training_set, columns=categorical_cols)
    testing_set = pd.get_dummies(testing_set, columns=categorical_cols)
    
    # Ensure testing set has all columns present in training set
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
    #endregion
    
    # Neural Network Parameters
    inputs = training_set.shape[1]
    num_hidden_nodes = 4
    learning_rate = 0.01
    iterations = 100
    
    hidden_nodes = []
    for _ in range(num_hidden_nodes):
        weight = np.random.uniform(-1, 1, size=inputs)
        bias = np.random.uniform(-1, 1)
        hidden_nodes.append(HiddenNode(weight, bias))
    
    output_weight = np.random.uniform(-1, 1, size=num_hidden_nodes)
    output_bias = np.random.uniform(-1, 1)  
    output_node = OutputNode(output_weight, output_bias)
    
    for epoch in range(iterations):
        # Shuffle training data and labels together each epoch
        shuffled_indices = np.random.permutation(len(training_set))
        training_set_shuffled = training_set.iloc[shuffled_indices].reset_index(drop=True)
        y_train_shuffled = y_train.iloc[shuffled_indices].reset_index(drop=True)

        for i in range(len(training_set_shuffled)):
            x = training_set_shuffled.iloc[i].values
            y = y_train_shuffled.iloc[i]

            hidden_outputs = []
            for node in hidden_nodes:
                hidden_outputs.append(node.calc_output(x))
            hidden_outputs = np.array(hidden_outputs)

            output = output_node.calc_output(hidden_outputs)
            output_error = output_node.calc_error(y)

            for j, node in enumerate(hidden_nodes):
                downstream_weight = output_node.weight[j]
                node.calc_error(downstream_weight, output_error)

            output_node.update_weights(learning_rate)
            for node in hidden_nodes:
                node.update_weights(learning_rate)

    
    results = []
    for i in range(len(testing_set)):
        x = testing_set.iloc[i].values
        hidden_outputs = []
        for node in hidden_nodes:
            hidden_outputs.append(node.calc_output(x))
        hidden_outputs = np.array(hidden_outputs)
        output = output_node.calc_output(hidden_outputs)
        pred = 1 if output >= 0.5 else 0
        results.append(pred)
    
    y_test_values = y_test.values
    accuracy = calc_accuracy(y_test_values, results)
    
    for pid, pred in zip(patient_ids_test, results):
        print(f"Patient {pid}: {'ckd' if pred == 1 else 'notckd'}")
    print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    main()