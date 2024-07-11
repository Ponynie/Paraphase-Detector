import os
import random
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

random.seed(69)
nltk.download('punkt')
nltk.download('wordnet')

def ensure_relative_path(path):
    current_dir = os.path.dirname(__file__)
    return os.path.join(current_dir, '..', path)

class SbertDataModule:
    
    def __init__(self, train_dir, test_dir, backTranslation_dir, ppdb_dir):
        self.sbert_model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        self.train_data = self._load_data(ensure_relative_path(train_dir), tag='train')
        self.test_data = self._load_data(ensure_relative_path(test_dir), tag='test')
        self.bt_data = self._load_data(ensure_relative_path(backTranslation_dir), tag='backTranslation')
        self.ppdb = self._load_ppdb(ensure_relative_path(ppdb_dir))
        
    def _load_data(self, filepath, tag):
        clean_data = {'Quality': [], '#1 ID': [], '#2 ID': [], '#1 String': [], '#2 String': [], 'Features': []}
        total_lines = sum(1 for _ in open(filepath, 'r'))
        with open(filepath, 'r') as file:
            next(file)  # Skip the header line
            for line in tqdm(file, total=total_lines - 1, desc=f"Loading, Embedding, Creating Features for {tag} data"):
                parts = line.strip().split('\t')
                if len(parts) == 5:
                    clean_data['Quality'].append(parts[0])
                    clean_data['#1 ID'].append(parts[1])
                    clean_data['#2 ID'].append(parts[2])
                    clean_data['#1 String'].append(parts[3])
                    clean_data['#2 String'].append(parts[4])
                    clean_data['Features'].append(np.maximum(self._get_sentence_vector(parts[3]), self._get_sentence_vector(parts[4])))
                else:
                    print(f"Skipping malformed line: {line}")
        return pd.DataFrame(clean_data)

    def _get_sentence_vector(self, sentence):
        return self.sbert_model.encode([sentence])[0]
    
    def _load_ppdb(self, ppdb_dir):
        ppdb = {}
        with open(ensure_relative_path(ppdb_dir), 'r') as f:
            for line in tqdm(f, desc="Loading PPDB"):
                word1, word2, _ = line.strip().split('\t')
                if word1 not in ppdb:
                    ppdb[word1] = []
                ppdb[word1].append(word2)
        return ppdb
    
    def get_augmented_train_data(self, p=0.25):
        # Start with the original training data
        augmented_data = self.train_data.copy()
        
        # Add backtranslation data
        augmented_data = pd.concat([augmented_data, self.bt_data], ignore_index=True)
        
        # Create new augmented data using synonym substitution
        synonym_augmented = []
        for _, row in tqdm(self.train_data.iterrows(), total=len(self.train_data), desc=f"Augmenting with synonym substitution with p = {p}"):
            s1_aug = self._synonym_substitution(row['#1 String'], p)
            s2_aug = self._synonym_substitution(row['#2 String'], p)
            
            if s1_aug != row['#1 String'] or s2_aug != row['#2 String']:
                new_features = np.maximum(self._get_sentence_vector(s1_aug), self._get_sentence_vector(s2_aug))
                synonym_augmented.append({
                    'Quality': row['Quality'],
                    '#1 ID': row['#1 ID'],
                    '#2 ID': row['#2 ID'],
                    '#1 String': s1_aug,
                    '#2 String': s2_aug,
                    'Features': new_features
                })
        
        # Create new augmented data using word paraphrasing
        paraphrase_augmented = []
        for _, row in tqdm(self.train_data.iterrows(), total=len(self.train_data), desc=f"Augmenting with word paraphrasing with p = {p}"):
            s1_aug = self._word_paraphrase(row['#1 String'], p)
            s2_aug = self._word_paraphrase(row['#2 String'], p)
            
            if s1_aug != row['#1 String'] or s2_aug != row['#2 String']:
                new_features = np.maximum(self._get_sentence_vector(s1_aug), self._get_sentence_vector(s2_aug))
                paraphrase_augmented.append({
                    'Quality': row['Quality'],
                    '#1 ID': row['#1 ID'],
                    '#2 ID': row['#2 ID'],
                    '#1 String': s1_aug,
                    '#2 String': s2_aug,
                    'Features': new_features
                })
        
        # Combine all augmented data
        augmented_data = pd.concat([
            augmented_data,
            pd.DataFrame(synonym_augmented),
            pd.DataFrame(paraphrase_augmented)
        ], ignore_index=True)
        
        return augmented_data
        
    def get_test_data(self):
        return self.test_data

    def _synonym_substitution(self, sentence, p=0.25):
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
    
    def _word_paraphrase(self, sentence, p=0.25):  
        words = word_tokenize(sentence)
        new_words = []
        for word in words:
            if random.random() < p and word in self.ppdb:
                paraphrase = random.choice(self.ppdb[word])
                new_words.append(paraphrase)
            else:
                new_words.append(word)
        return ' '.join(new_words)
    
class SVMModel:
    def __init__(self, kernel='rbf'):
        self.model = None
        self.kernel = kernel

    def train(self, data_df: pd.DataFrame, C=1.0, gamma='scale'):
        X = np.stack(data_df['Features'].to_numpy())
        y = data_df['Quality'].to_numpy()

        self.model = SVC(kernel=self.kernel, C=C, gamma=gamma)
        self.model.fit(X, y)

    def test(self, data_df: pd.DataFrame):
        X = np.stack(data_df['Features'].to_numpy())
        y = data_df['Quality'].to_numpy()

        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred)

        return accuracy

    def train_and_test(self, train_df, test_df, C=1.0, gamma='scale'):
        self.train(train_df, C, gamma)
        accuracy = self.test(test_df)
        return accuracy


