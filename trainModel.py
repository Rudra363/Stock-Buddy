import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib


def load_data(filepath="training_data.csv"):
    """
    Load dataset from CSV file and separate features and labels.
    """
    df = pd.read_csv(filepath)
    X = df.drop("label", axis=1)
    y = df["label"]
    return X, y

def train_model(X, y, test_size=0.2, random_state=42):
    """
    Split data, train Random Forest model, return trained model and test data.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    """
    Predict using model on test data and print accuracy and classification report.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))


def save_model(model, filepath="stock_label_model.pkl"):
    """
    Save the trained model to a file.
    """
    joblib.dump(model, filepath)
    print(f"Model saved as '{filepath}'")


def load_model(filepath="stock_label_model.pkl"):
    """
    Load a trained model from a file.
    """
    return joblib.load(filepath)

if __name__ == "__main__":
    X, y = load_data("training_data.csv")
    model, X_test, y_test = train_model(X, y)
    evaluate_model(model, X_test, y_test)
    save_model(model, "stock_label_model.pkl")