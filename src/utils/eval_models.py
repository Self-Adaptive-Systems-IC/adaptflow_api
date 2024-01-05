from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score


def evatuale_performance(x_train, y_train, x_test, y_test, classifier):
    prediction = classifier.predict(x_test)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    acc = accuracy_score(y_test, prediction)
    cross_val = cross_val_score(
        classifier, x_train, y_train, cv=cv, scoring="roc_auc"
    ).mean()
    roc_score = roc_auc_score(y_test, prediction)
    return (acc, cross_val, roc_score)
