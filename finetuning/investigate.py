from module import SbertDataModule, SVMModel, BertDataModule, train_fnn_model
import csv
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def reduce():    
    
    # Initialize data module
    data_module = SbertDataModule(
        train_dir='finetuning/MRPC_train.txt',
        test_dir='finetuning/MRPC_test.txt',
        backTranslation_dir='finetuning/MRPC_bt.txt',
        ppdb_dir='finetuning/PPDB-2.0-lexical.txt'
    )
    test_data = data_module.get_test_data()

    # Initialize SVM model
    svm_model = SVMModel()

    # Define the CSV file name
    csv_file = "finetuning/reduce_results.csv"

    # # Write the header to the CSV file (if it doesn't exist)
    # with open(csv_file, mode='a', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["Description", "Accuracy"])

    # Test with reduced dataset 
    print("\nTesting with reduced dataset (25%):")
    reduced_train_data = data_module.get_train_data(reduction_factor=0.25)
    reduced_accuracy = svm_model.train_and_test(reduced_train_data, test_data)
    print(f"Reduced dataset accuracy: {reduced_accuracy}")

    # Write the result to the CSV file
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Reduced dataset (25%)", reduced_accuracy])

    config = [('both', False), ('both', True), ('paraphrase', False), ('paraphrase', True), ('synonym', False), ('synonym', True)]
    for augment_method, include_bt in config:
        # Test with augmented dataset (Reduced)
        print(f"\nTesting with augmented dataset (25%) using {augment_method} augmentation method and include_bt={include_bt}:")
        augmented_train_data = data_module.get_augmented_train_data(p=0.7, include_bt=include_bt, reduction_factor=0.25, augment_method=augment_method)
        augmented_accuracy = svm_model.train_and_test(augmented_train_data, test_data)
        print(f"Augmented dataset accuracy: {augmented_accuracy}")

        # Write the result to the CSV file
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([f"Augmented dataset (25%) {augment_method}, include_bt={include_bt}", augmented_accuracy])
    return data_module

def neural(module=None):
    
    if module is None:
        # Initialize data module
        data_module = SbertDataModule(
            train_dir='finetuning/MRPC_train.txt',
            test_dir='finetuning/MRPC_test.txt',
            backTranslation_dir='finetuning/MRPC_bt.txt',
            ppdb_dir='finetuning/PPDB-2.0-lexical.txt'
        )
    else:
        data_module = module

    # Get test data
    test_data = data_module.get_test_data()

    # Test with full augmented dataset
    print("Testing with full augmented dataset:")
    full_train_data = data_module.get_augmented_train_data(p=0.7, include_bt=True, augment_method='both')

    # Get unaugmented train data
    unaugmented_train_data = data_module.get_train_data()

    # FNN Model parameters
    input_size = full_train_data['Features'].iloc[0].shape[0]
    hidden_size = 64
    output_size = len(full_train_data['Quality'].unique())

    # Train and test with augmented data
    fnn_model_augmented, full_fnn_result_augmented = train_fnn_model(full_train_data, test_data, input_size, hidden_size, output_size)
    print(f"Full augmented dataset FNN test result: {full_fnn_result_augmented}")

    # Train and test with unaugmented data
    fnn_model_unaugmented, full_fnn_result_unaugmented = train_fnn_model(unaugmented_train_data, test_data, input_size, hidden_size, output_size)
    print(f"Unaugmented dataset FNN test result: {full_fnn_result_unaugmented}")

    # Save the results to CSV
    result_df = pd.DataFrame({
        'Augmented': full_fnn_result_augmented,
        'Unaugmented': full_fnn_result_unaugmented
    })
    result_df.to_csv('finetuning/nn_result.csv', index=False)

def bert():
    # Initialize data module
    data_module = BertDataModule(
        train_dir='finetuning/MRPC_train.txt',
        test_dir='finetuning/MRPC_test.txt',
        backTranslation_dir='finetuning/MRPC_bt.txt',
        ppdb_dir='finetuning/PPDB-2.0-lexical.txt'
    )
    # Get test data
    test_data = data_module.get_test_data()
    
    # Initialize SVM model
    svm_model = SVMModel()
    
    # Define the CSV file name
    csv_file = "finetuning/bert_results.csv"

    # # Write the header to the CSV file (if it doesn't exist)
    # with open(csv_file, mode='a', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["Description", "Accuracy"])

    # Test with full unaugmented dataset 
    print("\nTesting with full unaugmented dataset:")
    full_train_data = data_module.get_train_data()
    accuracy = svm_model.train_and_test(full_train_data, test_data)
    print(f"Full unaugmented dataset BERT accuracy: {accuracy}")

    # Write the result to the CSV file
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Full unaugmented dataset BERT", accuracy])
    
def find_bad_aug():
    
    # Initialize data module
    data_module = SbertDataModule(
        train_dir='finetuning/MRPC_train.txt',
        test_dir='finetuning/MRPC_test.txt',
        backTranslation_dir='finetuning/MRPC_bt.txt',
        ppdb_dir='finetuning/PPDB-2.0-lexical.txt'
    )
    
    symnonym_aug = data_module.get_augmented_train_data(p=0.7, include_bt=False, augment_method='synonym')
    symnonym_aug.drop(['Features'], axis=1, inplace=True)
    symnonym_aug.to_csv('finetuning/view/synonym_aug.csv', index=False)
    
    paraphrase_aug = data_module.get_augmented_train_data(p=0.7, include_bt=False, augment_method='paraphrase')
    paraphrase_aug.drop(['Features'], axis=1, inplace=True)
    paraphrase_aug.to_csv('finetuning/view/paraphrase_aug.csv', index=False)
    
    original = data_module.get_train_data()
    original.to_csv('finetuning/view/original.csv', index=False)

def compare():
    # Initialize data module
    data_module = SbertDataModule(
        train_dir='finetuning/MRPC_train.txt',
        test_dir='finetuning/MRPC_test.txt',
        backTranslation_dir='finetuning/MRPC_bt.txt',
        ppdb_dir='finetuning/PPDB-2.0-lexical.txt'
    )

    # Get unaugmented training data
    unaugmented_train_data = data_module.get_train_data()

    # Get augmented training data (p=0.7, both methods, including backtranslation)
    augmented_train_data = data_module.get_augmented_train_data(p=0.7, augment_method='both', include_bt=True)

    # Get test data
    test_data = data_module.get_test_data()

    # Initialize and train SVM models
    svm_unaugmented = SVMModel()
    svm_augmented = SVMModel()

    svm_unaugmented.train(unaugmented_train_data)
    svm_augmented.train(augmented_train_data)

    # Prepare test data for prediction
    X_test = np.stack(test_data['Features'].to_numpy())
    y_test = test_data['Quality'].to_numpy()

    # Make predictions
    y_pred_unaugmented = svm_unaugmented.model.predict(X_test)
    y_pred_augmented = svm_augmented.model.predict(X_test)

    # Create DataFrame with results
    results_df = pd.DataFrame({
        'Sentence1': test_data['#1 String'],
        'Sentence2': test_data['#2 String'],
        'Ground Truth': y_test,
        'Unaugmented Prediction': y_pred_unaugmented,
        'Augmented Prediction': y_pred_augmented
    })

    # Filter for cases where predictions differ
    diff_predictions = results_df[results_df['Unaugmented Prediction'] != results_df['Augmented Prediction']]

    # Calculate accuracies
    unaugmented_accuracy = accuracy_score(y_test, y_pred_unaugmented)
    augmented_accuracy = accuracy_score(y_test, y_pred_augmented)

    print(f"Unaugmented Model Accuracy: {unaugmented_accuracy:.4f}")
    print(f"Augmented Model Accuracy: {augmented_accuracy:.4f}")
    print(f"Number of differing predictions: {len(diff_predictions)}")

    # Save results to CSV
    diff_predictions.to_csv('finetuning/view/differing_predictions.csv', index=False)

    return diff_predictions, unaugmented_accuracy, augmented_accuracy

def synset_stopword_compare():
    
    data_module = SbertDataModule(
    train_dir='finetuning/MRPC_train.txt',
    test_dir='finetuning/MRPC_test.txt',
    backTranslation_dir='finetuning/MRPC_bt.txt',
    ppdb_dir='finetuning/PPDB-2.0-lexical.txt'
    )
    p=0.7

    # Get unaugmented training data
    unaugmented_train_data = data_module.get_train_data()

    # Get augmented training data (only synonym substitution)
    augmented_train_data = data_module.get_augmented_train_data(p=p, reduction_factor=1.0, include_bt=False, augment_method='synonym')

    # Get test data
    test_data = data_module.get_test_data()

    # Initialize and train SVM models
    svm_unaugmented = SVMModel()
    svm_augmented = SVMModel()

    svm_unaugmented.train(unaugmented_train_data)
    svm_augmented.train(augmented_train_data)

    # Prepare test data for prediction
    X_test = np.stack(test_data['Features'].to_numpy())
    y_test = test_data['Quality'].to_numpy()

    # Make predictions
    y_pred_unaugmented = svm_unaugmented.model.predict(X_test)
    y_pred_augmented = svm_augmented.model.predict(X_test)

    # Calculate metrics
    metrics = {
        'Model': ['Unaugmented', 'Augmented (Synonym)'],
        'Accuracy': [
            accuracy_score(y_test, y_pred_unaugmented),
            accuracy_score(y_test, y_pred_augmented)
        ],
        'Precision': [
            precision_score(y_test, y_pred_unaugmented, average='weighted'),
            precision_score(y_test, y_pred_augmented, average='weighted')
        ],
        'Recall': [
            recall_score(y_test, y_pred_unaugmented, average='weighted'),
            recall_score(y_test, y_pred_augmented, average='weighted')
        ],
        'F1-Score': [
            f1_score(y_test, y_pred_unaugmented, average='weighted'),
            f1_score(y_test, y_pred_augmented, average='weighted')
        ]
    }

    # Create DataFrame with results
    results_df = pd.DataFrame({
        'Sentence1': test_data['#1 String'],
        'Sentence2': test_data['#2 String'],
        'Ground Truth': y_test,
        'Unaugmented Prediction': y_pred_unaugmented,
        'Augmented Prediction': y_pred_augmented
    })

    # Filter for cases where predictions differ
    diff_predictions = results_df[results_df['Unaugmented Prediction'] != results_df['Augmented Prediction']]

    # Print summary
    print(f"Unaugmented Model Accuracy: {metrics['Accuracy'][0]:.4f}")
    print(f"Augmented Model Accuracy: {metrics['Accuracy'][1]:.4f}")
    print(f"Number of differing predictions: {len(diff_predictions)}")

    # Save results to CSV
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv('finetuning/results/synonym_result.csv', index=False)
    # diff_predictions.to_csv('finetuning/view/synonym_stopword_differing_predictions.csv', index=False)
    
    # augmented_train_data.drop(['Features'], axis=1, inplace=True)
    # augmented_train_data.to_csv('finetuning/view/synonym_stopword_aug.csv', index=False)

    return metrics_df, diff_predictions


if __name__ == "__main__":
    synset_stopword_compare()
    