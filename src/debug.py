

augmented_datasets = [
    ('original', 'O'), #! Do not remove this (Original data needed to cache the augmented data)
    ('augmented_1', 'A1'), # Synonym Substitution
    ('augmented_2', 'A2'), # Word Paraphrase
    ('augmented_3', 'A3'), # BackTranslation
    ('augmented_4', 'A4'), # Random Word Deletion
    ('augmented_5', 'A5'), # Subject Object Switch
    ('full_augmented', 'FA') # All augmentations
]
import os

path = 'data/MRPC_data'
for folder, prefix in augmented_datasets:
    path_folder = os.path.join(path, folder)
    path_file = os.path.join(path_folder, f'{prefix}_MRPC_train.txt')
    with open(path_file, 'r') as file:
        print(f"File: {path_file}")
        f = file.readlines()
        length = len(f) - 1
        print(f"Number of lines: {length}")