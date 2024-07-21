from module import SbertDataModule, SVMModel, BertDataModule, train_fnn_model
import csv
import pandas as pd

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

    # Test with full dataset
    print("Testing with full augmented dataset:")
    full_train_data = data_module.get_augmented_train_data(p=0.7, include_bt=True, augment_method='both')
    
    # FNN Model
    input_size = full_train_data['Features'].iloc[0].shape[0]
    hidden_size = 64
    output_size = len(full_train_data['Quality'].unique())
    fnn_model, full_fnn_result = train_fnn_model(full_train_data, test_data, input_size, hidden_size, output_size)
    print(f"Full augmented dataset FNN test result: {full_fnn_result}")
    # Save the result to CSV
    result_df = pd.DataFrame(full_fnn_result)
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

    # Write the header to the CSV file (if it doesn't exist)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Description", "Accuracy"])

    # Test with reduced dataset 
    print("\nTesting with full augmented dataset:")
    full_train_data = data_module.get_augmented_train_data(p=0.7, include_bt=True, augment_method='both')
    accuracy = svm_model.train_and_test(full_train_data, test_data)
    print(f"Full augmented dataset BERT accuracy: {accuracy}")

    # Write the result to the CSV file
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Full augmented dataset BERT", accuracy])
    

if __name__ == "__main__":
    neural()
    bert()
    