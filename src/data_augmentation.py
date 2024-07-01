import os
import random
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from BackTranslation import BackTranslation
import spacy
from tqdm import tqdm
from time import sleep

def ensure_relative_path(path):
    current_dir = os.path.dirname(__file__)
    return os.path.join(current_dir, '..', path)

random.seed(42)

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Load spaCy model
spc = spacy.load('en_core_web_sm')

# Initialize backtranslation
bt = BackTranslation()

# Load PPDB
ppdb = {}
with open(ensure_relative_path('src/PPDB-2.0-lexical.txt'), 'r') as f:
    for line in tqdm(f, desc="Loading PPDB"):
        word1, word2, _ = line.strip().split('\t')
        if word1 not in ppdb:
            ppdb[word1] = []
        ppdb[word1].append(word2)

def synonym_substitution(sentence, p=0.25):
    words = word_tokenize(sentence)
    new_words = []
    for word in words:
        if random.random() < p:
            synsets = wordnet.synsets(word)
            if synsets:
                synonym = random.choice(synsets).lemmas()[0].name() #? what?
                new_words.append(synonym)
            else:
                new_words.append(word)
        else:
            new_words.append(word)
    return ' '.join(new_words)

def word_paraphrase(sentence, p=0.25):
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
    result = bt.translate(sentence, src='en', tmp='de', sleeping=1)
    return result.result_text

def backtranslation(sentence, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = bt.translate(sentence, src='en', tmp='de', sleeping=1)
            return result.result_text
        except Exception as e:
            print(f"Backtranslation failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
            sleep(2 ** attempt)  # Exponential backoff
    print(f"Backtranslation failed after {max_retries} attempts")
    return sentence  # Return original sentence if all attempts fail

def random_word_deletion(sentence, p=0.25):
    words = word_tokenize(sentence)
    if len(words) == 1:
        return sentence
    new_words = [word for word in words if random.random() > p]
    if len(new_words) == 0:
        return random.choice(words)
    return ' '.join(new_words)

def subject_object_switch(sentence):
    doc = spc(sentence)
    subject = None
    object = None
    verb = None

    for token in doc:
        if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
            subject = token
            verb = token.head
        elif token.dep_ == "dobj" and token.head == verb:
            object = token

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
        header = infile.readline()
        outfile.write(header)
        
        lines = infile.readlines()
        for line in tqdm(lines, desc=f"Augmenting with {augmentation_method}"):
            parts = line.strip().split('\t')
            if len(parts) == 5:
                quality, id1, id2, s1, s2 = parts
                
                # Generate and write augmented row
                if augmentation_method == 'synonym_substitution':
                    s1_aug = synonym_substitution(s1)
                    s2_aug = synonym_substitution(s2)
                    id1 = f"{id1}_A1"; id2 = f"{id2}_A1"
                elif augmentation_method == 'word_paraphrase':
                    s1_aug = word_paraphrase(s1)
                    s2_aug = word_paraphrase(s2)
                    id1 = f"{id1}_A2"; id2 = f"{id2}_A2"
                elif augmentation_method == 'backtranslation': 
                    s1_aug = backtranslation(s1)
                    s2_aug = backtranslation(s2)
                    id1 = f"{id1}_A3"; id2 = f"{id2}_A3"
                elif augmentation_method == 'random_word_deletion' and quality == '1':
                    s1_aug = random_word_deletion(s1)
                    s2_aug = random_word_deletion(s2)
                    id1 = f"{id1}_A4"; id2 = f"{id2}_A4"
                    quality = '0' # Set quality to 0 for random word deletion
                elif augmentation_method == 'subject_object_switch' and quality == '1':
                    if random.random() < 0.5:
                        s1_aug = subject_object_switch(s1)
                        if s1_aug == s1:
                            s2_aug = subject_object_switch(s2)
                            s1_aug = s1
                        else:
                            s2_aug = s2
                    else:
                        s2_aug = subject_object_switch(s2)
                        if s2_aug == s2:
                            s1_aug = subject_object_switch(s1)
                            s2_aug = s2
                        else:
                            s1_aug = s1
                    id1 = f"{id1}_A5"; id2 = f"{id2}_A5"
                    quality = '0'  # Set quality to 0 for subject-object switch
                else:
                    continue
                
                if s1_aug != s1 or s2_aug != s2:
                    outfile.write(f"{quality}\t{id1}\t{id2}\t{s1_aug}\t{s2_aug}\n")
            else:
                print(f"Skipping malformed line: {line}")

def process_dataset(dataset_type, base_data_dir):
    original_train = os.path.join(base_data_dir, dataset_type, 'original', f'O_{dataset_type.split("_")[0]}_train.txt')

    # augmentation_methods = [
    #     ('augmented_1', 'A1', 'synonym_substitution'),
    #     ('augmented_2', 'A2', 'word_paraphrase'),
    #     ('augmented_3', 'A3', 'backtranslation'),
    #     ('augmented_4', 'A4', 'random_word_deletion'),
    #     ('augmented_5', 'A5', 'subject_object_switch')
    # ]
    
    augmentation_methods = [
        ('augmented_1', 'A1', 'synonym_substitution'),
        ('augmented_2', 'A2', 'word_paraphrase'),
        ('augmented_4', 'A4', 'random_word_deletion'),
        ('augmented_5', 'A5', 'subject_object_switch')
    ]

    for folder, prefix, method in augmentation_methods:
        print(f"Processing {method} for {dataset_type}")
        
        output_folder = os.path.join(base_data_dir, dataset_type, folder)
        os.makedirs(output_folder, exist_ok=True)
        augmented_train = os.path.join(output_folder, f'{prefix}_{dataset_type.split("_")[0]}_train.txt')

        augment_data(original_train, augmented_train, method)

    # Create full augmented dataset
    print(f"Creating full augmented dataset for {dataset_type}")
    
    full_augmented_folder = os.path.join(base_data_dir, dataset_type, 'full_augmented')
    os.makedirs(full_augmented_folder, exist_ok=True)
    full_augmented_train = os.path.join(full_augmented_folder, f'FA_{dataset_type.split("_")[0]}_train.txt')

    with open(full_augmented_train, 'w') as outfile:
        # Write header
        with open(original_train, 'r') as infile:
            outfile.write(infile.readline())
        
        # Write all augmented data
        for folder, prefix, _ in tqdm(augmentation_methods, desc="Combining augmented datasets"):
            with open(os.path.join(base_data_dir, dataset_type, folder, f'{prefix}_{dataset_type.split("_")[0]}_train.txt'), 'r') as infile:
                next(infile)  # Skip header
                outfile.write(infile.read())
    
    print(f"Completed processing for {dataset_type}\n")

def main():
    base_data_dir = 'data'
    base_data_dir = ensure_relative_path(base_data_dir)
    
    # process_dataset('Sample_data', base_data_dir)
    process_dataset('MRPC_data', base_data_dir)

if __name__ == "__main__":
    main()