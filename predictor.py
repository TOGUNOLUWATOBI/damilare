import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
import tensorflow as tf

def predict_symptom_knn(symptom_array):
    train_df = pd.read_csv("./dataset/Training.csv")
    test_df = pd.read_csv("./dataset/Testing.csv")

    x = train_df.drop(["prognosis"], axis=1)
    x_test = test_df.drop(["prognosis"], axis=1)
    y = train_df["prognosis"]
    y_test = test_df["prognosis"]

    # Handle missing values by replacing NaN with mean values of respective columns
    imputer = SimpleImputer(strategy='mean')
    x = imputer.fit_transform(x)
    x_test = imputer.fit_transform(x_test)

    # Create a KNN classifier with k=3
    knn = KNeighborsClassifier(n_neighbors=3)

    # Train the classifier
    knn.fit(x, y)
    symptom_array = np.array(symptom_array)

    nn = NearestNeighbors(n_neighbors=10)
    nn.fit(x)
    _, indices = nn.kneighbors(symptom_array)

    # nearest_neighbors = [[y[i] for i in neighbors] for neighbors in indices]

    # # Calculate the accuracy of the model
    # accuracy = accuracy_score(y_test, y_pred)
    # print("Accuracy:", accuracy)
    pred = knn.predict(symptom_array)

    return pred


def predict_symptom_svm(symptom_array):
    train_df = pd.read_csv("./dataset/Training.csv")
    test_df = pd.read_csv("./dataset/Testing.csv")

    x = train_df.drop(["prognosis"], axis=1)
    x_test = test_df.drop(["prognosis"], axis=1)
    y = train_df["prognosis"]
    y_test = test_df["prognosis"]

    # Handle missing values by replacing NaN with mean values of respective columns
    imputer = SimpleImputer(strategy='mean')
    x = imputer.fit_transform(x)
    x_test = imputer.fit_transform(x_test)

    # Split the data into training and test sets
    # X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

    # Create an SVM classifier
    svm = SVC()

    # Train the classifier
    svm.fit(x, y)

    symptom_array = np.array(symptom_array)
    # Make predictions on the test set
    y_pred = svm.predict(symptom_array)
    return y_pred
    # # Calculate the accuracy of the model


def predict_neural_network(symptom_array):
    train_df = pd.read_csv("./dataset/Training.csv")
    test_df = pd.read_csv("./dataset/Testing.csv")
    y = train_df["prognosis"]
    model = tf.keras.models.load_model("symptom.h5")
    symptom_array = np.array(symptom_array)
    y_pred_enc = model.predict(symptom_array)
    y_pred = pd.get_dummies(y).columns[np.argmax(y_pred_enc, axis=1)]
    return y_pred