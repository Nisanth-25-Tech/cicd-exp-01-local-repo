
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


class DecisionTreeAIML:
    def __init__(self, max_depth=3):
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=42
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        print("Decision Tree model trained successfully!")

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)

        print("Accuracy:",
              f"{accuracy_score(y_test, predictions) * 100:.2f}%")

        print("\nClassification Report:")
        print(classification_report(y_test, predictions))


if __name__ == "__main__":
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    classifier = DecisionTreeAIML(max_depth=3)
    classifier.train(X_train, y_train)
    classifier.evaluate(X_test, y_test)

    prediction = classifier.predict([X_test[0]])
    print("Predicted flower:",
          iris.target_names[prediction[0]])
