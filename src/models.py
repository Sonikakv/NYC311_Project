from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB

def get_classification_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(max_depth=10),
        "Random Forest": RandomForestClassifier(n_estimators=150),
        "Naive Bayes": GaussianNB(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

def get_regression_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100)
    }