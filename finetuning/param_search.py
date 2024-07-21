import os
import pandas as pd
from itertools import product
from tqdm import tqdm
from finetuning.module import SbertDataModule
from finetuning.module import SVMModel

# Paths to data files
TRAIN_DIR = 'finetuning/MRPC_train.txt'
TEST_DIR = 'finetuning/MRPC_test.txt'
BACKTRANSLATION_DIR = 'finetuning/MRPC_bt.txt'
PPDB_DIR = 'finetuning/PPDB-2.0-lexical.txt'

# Hyperparameters to search
P_VALUES = [0.35, 0.5, 0.7, 0.8]
C_VALUES = [0.1, 1, 10]
GAMMA_VALUES = ['scale']
#GAMMA_VALUES = ['scale', 'auto', 0.1, 1]

# Results file
RESULTS_FILE = 'finetuning/param_search_results.csv'

def load_existing_results():
    if os.path.exists(RESULTS_FILE):
        return pd.read_csv(RESULTS_FILE)
    return pd.DataFrame(columns=['p', 'C', 'gamma', 'accuracy'])

def save_result(result, results_df):
    results_df = pd.concat([results_df, pd.DataFrame([result])], ignore_index=True)
    results_df.to_csv(RESULTS_FILE, index=False)
    return results_df

def main():
    # Initialize data module
    data_module = SbertDataModule(TRAIN_DIR, TEST_DIR, BACKTRANSLATION_DIR, PPDB_DIR)

    # Initialize SVM model
    svm_model = SVMModel()

    # Load existing results
    results_df = load_existing_results()

    # Get all combinations of hyperparameters
    all_combinations = list(product(P_VALUES, C_VALUES, GAMMA_VALUES))

    # Filter out already tested combinations
    tested_combinations = set(zip(results_df['p'], results_df['C'], results_df['gamma']))
    combinations_to_test = [combo for combo in all_combinations if combo not in tested_combinations]

    # Hyperparameter search
    for p, C, gamma in tqdm(combinations_to_test, desc="Hyperparameter search"):
        # Get augmented training data
        train_data = data_module.get_augmented_train_data(p=p)
        test_data = data_module.get_test_data()

        # Train and test the model
        accuracy = svm_model.train_and_test(train_data, test_data, C=C, gamma=gamma)

        # Save result immediately
        result = {
            'p': p,
            'C': C,
            'gamma': gamma,
            'accuracy': accuracy
        }
        results_df = save_result(result, results_df)

    # Sort results by accuracy
    results_df = results_df.sort_values('accuracy', ascending=False)

    # Save final sorted results
    results_df.to_csv(RESULTS_FILE, index=False)

    # Print best results
    print("Best hyperparameters:")
    print(results_df.iloc[0])

if __name__ == "__main__":
    main()