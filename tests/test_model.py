# tests/test_model.py
import os
import joblib

def test_model_file_exists():
    assert os.path.exists("wwe_model.pkl")
    clf = joblib.load("wwe_model.pkl")
    # optionally assert it has the scikit-learn predict method
    assert hasattr(clf, "predict")
