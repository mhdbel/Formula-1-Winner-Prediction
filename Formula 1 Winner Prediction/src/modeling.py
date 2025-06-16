from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump

def train_model(X_train, y_train):
    """
    Train a Random Forest classifier.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model.
    """
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def save_model(model, filepath):
    """
    Save the trained model to a file.
    """
    dump(model, filepath)

if __name__ == '__main__':
    # Example usage
    processed_data = pd.read_csv('data/processed_data/processed_canadian_gp.csv')
    X = processed_data.drop(columns=['Position', 'Points', 'Win'])
    y = processed_data['Win']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model, 'models/random_forest_model.pkl')