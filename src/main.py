# main.py
import argparse
from pipeline import NLPPipeline
import os
import pandas as pd

def process_dataset(dataset_type, base_data_dir, base_results_dir, augmented_datasets):
    print(f"Processing {dataset_type}")
    
    for dataset, prefix in augmented_datasets:
        print(f"Processing augmented dataset: {dataset}")
        
        #! Be careful with the file path
        train_dir = os.path.join(base_data_dir, dataset_type, dataset, f'{prefix}_{dataset_type.split("_")[0]}_train.txt') #! TRAIN
        test_dir = os.path.join(base_data_dir, dataset_type, 'test_data', f'{dataset_type.split("_")[0]}_test.txt') #! TEST
        results_dir = os.path.join(base_results_dir, dataset_type, dataset)
        
        pipeline = NLPPipeline(train_dir, test_dir, results_dir, prefix=prefix, save_models=False)
        pipeline.execute() 
        
        print(f"Completed processing for {dataset_type} - {dataset}\n")
    
    # Reset the cache
    NLPPipeline.original_bow_cache = pd.DataFrame() 
    NLPPipeline.original_sbert_cache = pd.DataFrame()
    NLPPipeline.testset_bow_cache = pd.DataFrame()
    NLPPipeline.testset_sbert_cache = pd.DataFrame()
    print(f"Completed processing for {dataset_type}")
    

def main():
    #* Type python src/main.py --mode sample or python src/main.py --mode full to run the pipeline
    parser = argparse.ArgumentParser(description="Run NLP pipeline on sample or full data.")
    parser.add_argument('--mode', choices=['sample', 'full'], required=True,
                        help="'sample' to run on Sample data, 'full' to run on MRPC data")
    args = parser.parse_args()

    base_data_dir = 'data'
    base_results_dir = 'results'

    augmented_datasets = [
        ('original', 'O'), #! Do not remove this (Original data needed to cache the augmented data)
        ('augmented_1', 'A1'), # Synonym Substitution
        ('augmented_2', 'A2'), # Word Paraphrase
        ('augmented_3', 'A3'), # BackTranslation
        ('augmented_4', 'A4'), # Random Word Deletion
        ('augmented_5', 'A5'), # Subject Object Switch
        ('full_augmented', 'FA') # All augmentations
    ]

    if args.mode == 'sample':
        print("Running in sample mode with Sample data...")
        process_dataset('Sample_data', base_data_dir, base_results_dir, augmented_datasets)
        print("Sample mode completed successfully. The pipeline is working as expected.")
        print("You can now run the pipeline with real data using the 'full' mode.")
    elif args.mode == 'full':
        print("Running in full mode with MRPC data...")
        process_dataset('MRPC_data', base_data_dir, base_results_dir, augmented_datasets)
        print("Full run completed successfully.")

if __name__ == "__main__":
    main()