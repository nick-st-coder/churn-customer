import optuna
from lightgbm import LGBMClassifier, early_stopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def tune_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=86, stratify=y
    )

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=42
    )

    def objective(trial):

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 20, log=True),

            "subsample_freq": 1,
            "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
            "random_state": 87,
            "n_jobs": -1,
            "objective": "binary",
            "verbosity": -1
        }

        model = LGBMClassifier(**params)

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[early_stopping(50, verbose=False)]
        )

        proba = model.predict_proba(X_val)[:, 1]

        return roc_auc_score(y_val, proba)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Best Params:", study.best_params)
    return study.best_params