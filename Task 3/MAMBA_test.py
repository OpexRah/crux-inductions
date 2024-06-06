import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import tensorflow as tf

df = pd.read_csv("dataset/youtube_dataset_balanced.csv")
df = df.sample(frac=1).reset_index(drop=True)

texts = df['title']
labels = df['Category']

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
vocab_size = tokenizer.vocab_size
tokenized_texts = tokenizer(texts.tolist(), padding='max_length', truncation=True, max_length=36, return_tensors="np")

input_ids = tokenized_texts['input_ids']
labels = np.array(labels)

x_train, x_val, y_train, y_val = train_test_split(input_ids, labels, test_size=0.2, random_state=42)

BATCH_SIZE = 32
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(BATCH_SIZE).shuffle(1000)
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(BATCH_SIZE).shuffle(1000)