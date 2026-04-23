from sklearn.model_selection import GridSearchCV

def tune_model(model, param_grid, X_train, y_train):
    grid = GridSearchCV(
        model,
        param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("\nBest Parameters:", grid.best_params_)
    print("Best CV Score:", grid.best_score_)

    return grid.best_estimator_