import numpy as np

from decision_tree import DecisionTreeClassifier


def test_decision_tree_smoke():
    X_train = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )
    y_train = np.array([0, 0, 1, 1])

    model = DecisionTreeClassifier(max_depth=3)
    model.fit(X_train, y_train)

    X_test = np.array([[0.2, 0.8], [0.9, 0.2]])
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    assert preds.shape == (2,)
    assert probs.shape[0] == 2
    assert set(np.unique(preds)).issubset({0, 1})
    assert np.allclose(probs.sum(axis=1), 1.0)
