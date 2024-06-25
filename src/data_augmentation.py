import os
import random
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from BackTranslation import BackTranslation
import spacy

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Load PPDB
ppdb = {}
with open('PPDB-2.0-lexical.txt', 'r') as f:
    for line in f:
        word1, word2, _ = line.strip().split('\t')
        if word1 not in ppdb:
            ppdb[word1] = []
        ppdb[word1].append(word2)

def synonym_substitution(sentence, p=0.1):
    words = word_tokenize(sentence)
    new_words = []
    for word in words:
        if random.random() < p:
            synsets = wordnet.synsets(word)
            if synsets:
                synonym = random.choice(synsets).lemmas()[0].name()
                new_words.append(synonym)
            else:
                new_words.append(word)
        else:
            new_words.append(word)
    return ' '.join(new_words)

def word_paraphrase(sentence, p=0.1):
    words = word_tokenize(sentence)
    new_words = []
    for word in words:
        if random.random() < p and word in ppdb:
            paraphrase = random.choice(ppdb[word])
            new_words.append(paraphrase)
        else:
            new_words.append(word)
    return ' '.join(new_words)

def backtranslation(sentence):
    bt = BackTranslation(tmp='de', sleep_sec=1)
    result = bt.translate(sentence, src='en', tmp='de')
    return result.result_text

def random_word_deletion(sentence, p=0.1):
    words = word_tokenize(sentence)
    if len(words) == 1:
        return sentence
    new_words = [word for word in words if random.random() > p]
    if len(new_words) == 0:
        return random.choice(words)
    return ' '.join(new_words)

def subject_object_switch(sentence):
    doc = nlp(sentence)
    subject = None
    object = None
    verb = None
    
    for token in doc:
        if token.dep_ == "nsubj":
            subject = token
        elif token.dep_ == "dobj":
            object = token
        if token.pos_ == "VERB":
            verb = token
    
    if subject and object and verb:
        words = [token.text for token in doc]
        subject_index = subject.i
        object_index = object.i
        words[subject_index], words[object_index] = words[object_index], words[subject_index]
        return ' '.join(words)
    else:
        return sentence

def augment_data(input_file, output_file, augmentation_method):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        # Write header
        outfile.write(infile.readline())
        
        for line in infile:
            parts = line.strip().split('\t')
            if len(parts) == 5:
                quality, id1, id2, s1, s2 = parts
                
                if augmentation_method == 'synonym_substitution':
                    s1 = synonym_substitution(s1)
                    s2 = synonym_substitution(s2)
                elif augmentation_method == 'word_paraphrase':
                    s1 = word_paraphrase(s1)
                    s2 = word_paraphrase(s2)
                elif augmentation_method == 'backtranslation':
                    s1 = backtranslation(s1)
                    s2 = backtranslation(s2)
                elif augmentation_method == 'random_word_deletion':
                    s1 = random_word_deletion(s1)
                    s2 = random_word_deletion(s2)
                elif augmentation_method == 'subject_object_switch':
                    s1 = subject_object_switch(s1)
                    s2 = subject_object_switch(s2)
                
                outfile.write(f"{quality}\t{id1}\t{id2}\t{s1}\t{s2}\n")
            else:
                print(f"Skipping malformed line: {line}")

def ensure_relative_path(path):
    current_dir = os.path.dirname(__file__)
    return os.path.join(current_dir, '..', path)

def process_dataset(dataset_type, base_data_dir):
    original_train = os.path.join(base_data_dir, dataset_type, 'original', f'O_{dataset_type.split("_")[0]}_train.txt')
    original_test = os.path.join(base_data_dir, dataset_type, 'original', f'O_{dataset_type.split("_")[0]}_test.txt')

    augmentation_methods = [
        ('augmented_1', 'A1', 'synonym_substitution'),
        ('augmented_2', 'A2', 'word_paraphrase'),
        ('augmented_3', 'A3', 'backtranslation'),
        ('augmented_4', 'A4', 'random_word_deletion'),
        ('augmented_5', 'A5', 'subject_object_switch')
    ]

    for folder, prefix, method in augmentation_methods:
        print(f"Processing {method} for {dataset_type}")
        output_folder = os.path.join(base_data_dir, dataset_type, folder)
        output_folder = ensure_relative_path(output_folder)
        os.makedirs(output_folder, exist_ok=True)

        augmented_train = os.path.join(output_folder, f'{prefix}_{dataset_type.split("_")[0]}_train.txt')
        augmented_test = os.path.join(output_folder, f'{prefix}_{dataset_type.split("_")[0]}_test.txt')

        augment_data(original_train, augmented_train, method)
        augment_data(original_test, augmented_test, method)

    # Create full augmented dataset
    print(f"Creating full augmented dataset for {dataset_type}")
    full_augmented_folder = os.path.join(base_data_dir, dataset_type, 'full_augmented')
    full_augmented_folder = ensure_relative_path(full_augmented_folder)
    os.makedirs(full_augmented_folder, exist_ok=True)

    full_augmented_train = os.path.join(full_augmented_folder, f'FA_{dataset_type.split("_")[0]}_train.txt')
    full_augmented_test = os.path.join(full_augmented_folder, f'FA_{dataset_type.split("_")[0]}_test.txt')

    with open(full_augmented_train, 'w') as outfile:
        for folder, prefix, _ in augmentation_methods:
            with open(os.path.join(base_data_dir, dataset_type, folder, f'{prefix}_{dataset_type.split("_")[0]}_train.txt'), 'r') as infile:
                outfile.write(infile.read())

    with open(full_augmented_test, 'w') as outfile:
        for folder, prefix, _ in augmentation_methods:
            with open(os.path.join(base_data_dir, dataset_type, folder, f'{prefix}_{dataset_type.split("_")[0]}_test.txt'), 'r') as infile:
                outfile.write(infile.read())

def main():
    base_data_dir = 'data'
    
    process_dataset('testrun_data', base_data_dir)
    process_dataset('MRPC_data', base_data_dir)

if __name__ == "__main__":
    main()