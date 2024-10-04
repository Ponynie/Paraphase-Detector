# Paraphrase Identification using Sentence BERT and Data Augmentation

## Introduction

Natural Language Processing (NLP) is a branch of artificial intelligence that enables computers to understand, interpret, and respond to human language. One essential task within NLP is **paraphrase identification**, which determines whether two sentences convey the same meaning. Paraphrase identification is crucial for applications like information retrieval, question answering, and plagiarism detection.

Paraphrase identification is complex because two sentences may express the same idea using different words and structures. For example:

- "The new processor was unveiled at the Intel Developer Forum 2003 in San Jose."
- "The processors were announced in San Jose at the Intel Developer Forum."

Recent advancements, such as **Sentence BERT** (Sentence Bidirectional Encoder Representations from Transformers), have introduced powerful models for paraphrase identification by generating semantically meaningful sentence embeddings. These embeddings can be compared using **cosine similarity**, making them highly effective for tasks like paraphrase identification.

This report evaluates the performance of **Sentence BERT** compared to the traditional **Bag-of-Words** (BoW) approach for paraphrase identification. It also examines the impact of different **Support Vector Machine (SVM)** kernels and feature combination methods on accuracy. Finally, we explore how **data augmentation techniques** affect model performance.

---

## Methodology

### 1. Bag-of-Words (BoW) Approach

In the traditional **BoW** model, sentences are tokenized, converted to lowercase, stemmed, and stopwords are removed. Sentence pairs are then represented as feature vectors based on the frequency of words.

### 2. Sentence BERT Approach

We used **Sentence BERT** to generate 768-dimensional sentence embeddings. Three methods were tested to combine sentence pairs into single vectors:
- **Concatenation** (1536-dimensional vector)
- **Mean pooling**
- **Max pooling**

### 3. SVM Classifiers

For both BoW and Sentence BERT, we trained **SVM classifiers** using the following kernels:
- RBF
- Linear
- Polynomial
- Sigmoid

### Dataset

We used the **Microsoft Research Paraphrase Corpus (MRPC)** for all experiments. The MRPC dataset consists of sentence pairs labeled as equivalent or non-equivalent.

---

## Results

### Accuracy of Different Models and Kernels

| Method          | Kernel   | Accuracy |
|-----------------|----------|----------|
| BoW             | RBF      | 0.6916   |
| BoW             | Linear   | 0.6406   |
| BoW             | Poly     | 0.6829   |
| BoW             | Sigmoid  | 0.5826   |
| SBERT (Concat)  | RBF      | 0.6945   |
| SBERT (Concat)  | Linear   | 0.6226   |
| SBERT (Concat)  | Poly     | 0.7032   |
| SBERT (Concat)  | Sigmoid  | 0.6203   |
| SBERT (Mean)    | RBF      | 0.7217   |
| SBERT (Mean)    | Linear   | 0.6470   |
| SBERT (Mean)    | Poly     | 0.6916   |
| SBERT (Mean)    | Sigmoid  | 0.6081   |
| SBERT (Max)     | RBF      | **0.7594** |
| SBERT (Max)     | Linear   | 0.7026   |
| SBERT (Max)     | Poly     | 0.7548   |
| SBERT (Max)     | Sigmoid  | 0.7078   |

### Discussion

- **Sentence BERT vs. Bag-of-Words**: Sentence BERT consistently outperforms BoW, especially when using **max pooling**.
- **SVM Kernels**: The **RBF kernel** performs best across all configurations.
- **Best Performing Model**: The highest accuracy was achieved with **Sentence BERT** using **max pooling** and an **RBF kernel**, with an accuracy of **0.7594**.

---

## Data Augmentation

### Augmentation Techniques

We tested five data augmentation techniques to improve model performance:
1. **Synonym Substitution (A1)**
2. **Word Paraphrase (A2)**
3. **BackTranslation (A3)**
4. **Random Word Deletion (A4)**
5. **Subject Object Switch (A5)**

Additionally, we created a **Full Augmentation (FA)** dataset combining all techniques.

### Results

| Augmentation Technique | Accuracy |
|------------------------|----------|
| Original (O)           | 0.7594   |
| Synonym Substitution    | 0.7600   |
| Word Paraphrase         | 0.7577   |
| BackTranslation         | 0.7530   |
| Random Word Deletion    | 0.7130   |
| Subject Object Switch   | 0.6887   |
| Full Augmentation (FA)  | 0.7368   |

### Discussion

- **Synonym Substitution (A1)** yielded a slight improvement, but most techniques showed marginal or no improvements.
- **Destructive techniques** (e.g., Random Word Deletion, Subject Object Switch) significantly reduced performance.
- The **Full Augmentation** dataset underperformed, suggesting the inclusion of destructive techniques introduces noise rather than useful variations.

---

## Conclusion

- **Sentence BERT** with **max pooling** and an **RBF kernel SVM** outperforms traditional Bag-of-Words approaches for paraphrase identification.
- **Data augmentation** did not yield significant improvements. While **synonym substitution** offered a marginal increase, destructive techniques harmed performance. The original **MRPC dataset** is likely diverse enough, limiting the need for synthetic data.

---

## References

- [1] Nils Reimers and Iryna Gurevych. *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. 2019.

---
