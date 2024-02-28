import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import mlflow 
import optuna


# Path to train.py
file_path = os.path.dirname(__file__)

# Set experiments name
mlflow.set_experiment("Credit-card-experiments-tunning")

# Obtain hyperparameters for this trial
def suggest_hyperparameters(trial):

    # Obtain the max depth on a int format
    max_depth = trial.suggest_int('max_depth', 10, 50)

    # # Obtain the learning rate on a logarithmic scale
    # learning_rate = trial.suggest_float("lr", 1e-4, 1e-1, log=True)

    num_leaves = trial.suggest_int('num_leaves', 5, 25)

    return max_depth,num_leaves

def prepare_data_to_be_used():
    
    # Load data
    data_path = os.path.join(file_path, "..", "data", "UCI_Credit_Card.csv")
    df = pd.read_csv(data_path)

    # Get features and target name
    features = df.columns.to_list()[1:-1]
    target = df.columns.to_list()[-1]

    # Train test split
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=23)

    return df_train, df_test, features, target

def objective(trial):
    print("\n********************************\n")
   
    # Start a new mlflow run
    with mlflow.start_run():
        # Get hyperparameter suggestions created by optuna and log them as params using mlflow
        max_depth,num_leaves = suggest_hyperparameters(trial)
        mlflow.log_params(trial.params)

        df_train, df_test, features, target = prepare_data_to_be_used()
        # Train model
        mlflow.log_params(trial.params)
        clf = LGBMClassifier(**trial.params)
        clf.fit(df_train[features], df_train[target])

        # tracking do modelo
        signature = mlflow.models.infer_signature(df_train[features], clf.predict(df_train[features]))
        mlflow.sklearn.log_model(clf,
                                 "model_store",
                                 signature=signature,
                                 input_example=df_train[features].iloc[:2])

        # Evaluate
        gini_train = (
            2 * roc_auc_score(df_train[target], clf.predict_proba(df_train[features])[:, 1])
            - 1
        )
        gini_test = (
            2 * roc_auc_score(df_test[target], clf.predict_proba(df_test[features])[:, 1])
            - 1
        )

        # Tracking model metrics
        mlflow.log_metric("Gini_train", gini_train)
        mlflow.log_metric("Gini_test", gini_test)
   
    return 1 - gini_test

def main():
    # Create the optuna study which shares the experiment name
    study = optuna.create_study(study_name="sklearn-mlflow-optuna", direction="minimize")
    study.optimize(objective, n_trials=10)

    # Print optuna study statistics
    print("\n++++++++++++++++++++++++++++++++++\n")
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Trial number: ", trial.number)
    print("  Loss (trial value): ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    main()
