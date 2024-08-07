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
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torchmetrics
from transformers import BertTokenizer, BertModel

np.random.seed(69)
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
    
    def get_augmented_train_data(self, p=0.25, reduction_factor=1.0, include_bt=True, augment_method='both'):
        # Start with the original training data, reduced by reduction_factor
        reduced_train_data = self.train_data.sample(frac=reduction_factor, random_state=42)
        augmented_data = reduced_train_data.copy()
        
        # Add backtranslation data if requested
        if include_bt:
            augmented_data = pd.concat([augmented_data, self.bt_data], ignore_index=True)
        
        # Create new augmented data based on the specified method
        if augment_method in ['synonym', 'both']:
            synonym_augmented = self._augment_with_synonyms(reduced_train_data, p)
            augmented_data = pd.concat([augmented_data, pd.DataFrame(synonym_augmented)], ignore_index=True)
        
        if augment_method in ['paraphrase', 'both']:
            paraphrase_augmented = self._augment_with_paraphrases(reduced_train_data, p)
            augmented_data = pd.concat([augmented_data, pd.DataFrame(paraphrase_augmented)], ignore_index=True)
        
        return augmented_data
    
    def _augment_with_synonyms(self, data, p):
        synonym_augmented = []
        for _, row in tqdm(data.iterrows(), total=len(data), desc=f"Augmenting with synonym substitution with p = {p}"):
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
        return synonym_augmented

    def _augment_with_paraphrases(self, data, p):
        paraphrase_augmented = []
        for _, row in tqdm(data.iterrows(), total=len(data), desc=f"Augmenting with word paraphrasing with p = {p}"):
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
        return paraphrase_augmented
    
    def get_train_data(self, reduction_factor=1.0):
        return self.train_data.sample(frac=reduction_factor, random_state=42)
        
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
        print(f"{"-"*42}\nTraining SVM model with C={C}, gamma={gamma}\n{"-"*42}")
        X = np.stack(data_df['Features'].to_numpy())
        y = data_df['Quality'].to_numpy()

        self.model = SVC(kernel=self.kernel, C=C, gamma=gamma)
        self.model.fit(X, y)

    def test(self, data_df: pd.DataFrame):
        print(f"{"-"*20}\nTesting SVM model...\n{"-"*20}")
        X = np.stack(data_df['Features'].to_numpy())
        y = data_df['Quality'].to_numpy()

        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred)

        return accuracy

    def train_and_test(self, train_df, test_df, C=1.0, gamma='scale'):
        self.train(train_df, C, gamma)
        accuracy = self.test(test_df)
        return accuracy

class FNNModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        super().__init__()
        self.lr = learning_rate
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=output_size)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.log('train_loss', loss)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.log('test_loss', loss)
        self.log('test_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

class FNNDataModule(pl.LightningDataModule):
    def __init__(self, train_data, test_data, batch_size=16):
        super().__init__()
        self.train_data = train_data
        self.test_data = test_data
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Convert pandas DataFrames to PyTorch tensors
        self.X_train = torch.tensor(np.stack(self.train_data['Features'].to_numpy()), dtype=torch.float32)
        self.y_train = torch.tensor(self.train_data['Quality'].astype('category').cat.codes.to_numpy(), dtype=torch.long)
        
        self.X_test = torch.tensor(np.stack(self.test_data['Features'].to_numpy()), dtype=torch.float32)
        self.y_test = torch.tensor(self.test_data['Quality'].astype('category').cat.codes.to_numpy(), dtype=torch.long)

    def train_dataloader(self):
        train_dataset = TensorDataset(self.X_train, self.y_train)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        test_dataset = TensorDataset(self.X_test, self.y_test)
        return DataLoader(test_dataset, batch_size=self.batch_size)

def train_fnn_model(train_data, test_data, input_size, hidden_size, output_size, max_epochs=10):
    model = FNNModel(input_size, hidden_size, output_size)
    data_module = FNNDataModule(train_data, test_data)
    
    trainer = pl.Trainer(max_epochs=max_epochs, enable_checkpointing=False, logger=False)
    trainer.fit(model, data_module)
    
    test_result = trainer.test(model, data_module)
    return model, test_result


class BertDataModule:
    
    def __init__(self, train_dir, test_dir, backTranslation_dir, ppdb_dir, bert_model='bert-base-uncased', max_length=128):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.bert_model = BertModel.from_pretrained(bert_model)
        self.max_length = max_length
        self.train_data = self._load_data(ensure_relative_path(train_dir), tag='train')
        self.test_data = self._load_data(ensure_relative_path(test_dir), tag='test')
        self.bt_data = self._load_data(ensure_relative_path(backTranslation_dir), tag='backTranslation')
        self.ppdb = self._load_ppdb(ensure_relative_path(ppdb_dir))
        
    def _load_data(self, filepath, tag):
        clean_data = {'Quality': [], '#1 ID': [], '#2 ID': [], '#1 String': [], '#2 String': [], 'Features': []}
        total_lines = sum(1 for _ in open(filepath, 'r'))
        with open(filepath, 'r') as file:
            next(file)  # Skip the header line
            for line in tqdm(file, total=total_lines - 1, desc=f"Loading and Creating Features for {tag} data"):
                parts = line.strip().split('\t')
                if len(parts) == 5:
                    clean_data['Quality'].append(parts[0])
                    clean_data['#1 ID'].append(parts[1])
                    clean_data['#2 ID'].append(parts[2])
                    clean_data['#1 String'].append(parts[3])
                    clean_data['#2 String'].append(parts[4])
                    clean_data['Features'].append(self._get_bert_features(parts[3], parts[4]))
                else:
                    print(f"Skipping malformed line: {line}")
        return pd.DataFrame(clean_data)

    def _get_bert_features(self, sentence1, sentence2):
        inputs = self.tokenizer(sentence1, sentence2, return_tensors="pt", max_length=self.max_length, padding="max_length", truncation=True)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy().squeeze()  # Using [CLS] token as the feature vector
    
    def _load_ppdb(self, ppdb_dir):
        ppdb = {}
        with open(ensure_relative_path(ppdb_dir), 'r') as f:
            for line in tqdm(f, desc="Loading PPDB"):
                word1, word2, _ = line.strip().split('\t')
                if word1 not in ppdb:
                    ppdb[word1] = []
                ppdb[word1].append(word2)
        return ppdb
    
    def get_augmented_train_data(self, p=0.25, reduction_factor=1.0, include_bt=True, augment_method='both'):
        # Start with the original training data, reduced by reduction_factor
        reduced_train_data = self.train_data.sample(frac=reduction_factor, random_state=42)
        augmented_data = reduced_train_data.copy()
        
        # Add backtranslation data if requested
        if include_bt:
            augmented_data = pd.concat([augmented_data, self.bt_data], ignore_index=True)
        
        # Create new augmented data based on the specified method
        if augment_method in ['synonym', 'both']:
            synonym_augmented = self._augment_with_synonyms(reduced_train_data, p)
            augmented_data = pd.concat([augmented_data, pd.DataFrame(synonym_augmented)], ignore_index=True)
        
        if augment_method in ['paraphrase', 'both']:
            paraphrase_augmented = self._augment_with_paraphrases(reduced_train_data, p)
            augmented_data = pd.concat([augmented_data, pd.DataFrame(paraphrase_augmented)], ignore_index=True)
        
        return augmented_data
    
    def _augment_with_synonyms(self, data, p):
        synonym_augmented = []
        for _, row in tqdm(data.iterrows(), total=len(data), desc=f"Augmenting with synonym substitution with p = {p}"):
            s1_aug = self._synonym_substitution(row['#1 String'], p)
            s2_aug = self._synonym_substitution(row['#2 String'], p)
            
            if s1_aug != row['#1 String'] or s2_aug != row['#2 String']:
                new_features = self._get_bert_features(s1_aug, s2_aug)
                synonym_augmented.append({
                    'Quality': row['Quality'],
                    '#1 ID': row['#1 ID'],
                    '#2 ID': row['#2 ID'],
                    '#1 String': s1_aug,
                    '#2 String': s2_aug,
                    'Features': new_features
                })
        return synonym_augmented

    def _augment_with_paraphrases(self, data, p):
        paraphrase_augmented = []
        for _, row in tqdm(data.iterrows(), total=len(data), desc=f"Augmenting with word paraphrasing with p = {p}"):
            s1_aug = self._word_paraphrase(row['#1 String'], p)
            s2_aug = self._word_paraphrase(row['#2 String'], p)
            
            if s1_aug != row['#1 String'] or s2_aug != row['#2 String']:
                new_features = self._get_bert_features(s1_aug, s2_aug)
                paraphrase_augmented.append({
                    'Quality': row['Quality'],
                    '#1 ID': row['#1 ID'],
                    '#2 ID': row['#2 ID'],
                    '#1 String': s1_aug,
                    '#2 String': s2_aug,
                    'Features': new_features
                })
        return paraphrase_augmented
    
    def get_train_data(self, reduction_factor=1.0):
        return self.train_data.sample(frac=reduction_factor, random_state=42)
        
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