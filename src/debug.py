

augmentation_methods = [
    ('original', 'O', 'Original'),
    ('augmented_1', 'A1', 'Synonym Substitution'),
    ('augmented_2', 'A2', 'Word Paraphrase'),
    ('augmented_3', 'A3', 'Backtranslation'),
    ('augmented_4', 'A4', 'Random Word deletion'),
    ('augmented_5', 'A5', 'Subject Object switch')
]
import os

s = 0
a = 0
path = 'data/MRPC_data'
for folder, prefix, name in augmentation_methods:
    path_folder = os.path.join(path, folder)
    path_file = os.path.join(path_folder, f'{prefix}_MRPC_train.txt')
    with open(path_file, 'r') as file:
        print(f"Augmentation: {name}")
        f = file.readlines()
        length = len(f) - 1
        a += length
        print(f"Number of data: {length + s}")
        if prefix == 'O':
            s = length
            
print(f"Full Augmentation: {a}")