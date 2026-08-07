from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(model, X_test, y_test):

    preds = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("Recall Score:\n", recall_score(y_test, preds))
    print("Precision Score:\n", precision_score(y_test, preds))
    print("Acurracy Score:\n", accuracy_score(y_test, preds))
    print("F1 Score:\n", f1_score(y_test, preds))