import numpy as np
from src.metrics import classification_report, regression_report


def test_classification_report():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 0]
    report = classification_report(y_true, y_pred)
    assert "accuracy" in report
    assert 0 <= report["accuracy"] <= 1


def test_regression_report():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.1, 1.9, 3.2]
    report = regression_report(y_true, y_pred)
    assert "r2" in report
    assert "rmse" in report
