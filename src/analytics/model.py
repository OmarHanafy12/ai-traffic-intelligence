# model
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

FEATURES = [
    "vehicle_count",
    "cars",
    "motorcycles",
    "buses",
    "trucks",
    "in_count",
    "out_count",
    "density"
]

class TrafficModel:

    def __init__(self):

        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

    def train(self, df):

        X = df[FEATURES]
        y = df["traffic_level"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        self.model.fit(X_train, y_train)

        predictions = self.model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        print("Accuracy:", accuracy)

        return accuracy

    def predict(self, features):

        data = [[features[x] for x in FEATURES]]

        return self.model.predict(data)[0]

    def save(self, path):

        joblib.dump(self.model, path)

    def load(self, path):

        self.model = joblib.load(path)
