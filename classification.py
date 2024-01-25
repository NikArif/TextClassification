# %%
#1. Setup - Importing packages
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt
import sklearn
import os, pickle, datetime
from sklearn.metrics import f1_score, classification_report

# %%
# 2. Data loading
file = "ecommerceDataset.csv"
df = pd.read_csv(file, header=None, names=['category','text'])

# %%
# 3. Data Inspection
# df shape
print("Shape of the data: ",df.shape)

# %%
# df info
print("Data info: \n", df.info())

# %%
# df describe
print("Data description: \n", df.describe())

# %%
# df example
print("Data Example \n", df.head(1))

# %%
# 4. Data Cleaning
# Inspect null and duplicate Data Cleaning
print(df.isna().sum())
print("---------------------------------------------------")
print(df.duplicated().sum())

# %%
#removing null rows since there's only one rows
df = df.dropna()

# %%
# 5. Dealing with duplicates
categories_list = df['category'].unique()
print(df["category"].value_counts())

# %%
# 6. Removes duplicates and see the class representation again
df_no_dup = df.drop_duplicates()
print(df_no_dup["category"].value_counts())

# %%
# 7. Data preprocessing
# 7.1 Split the data into features and labels
features = df_no_dup['text'].values
labels = df_no_dup['category'].values

# %%
# 7.2 convert the categorical label into integer - label encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# %%
# 7.3 perform train_test_split
from sklearn.model_selection import train_test_split
seed =42
X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, train_size=0.8, random_state=seed)

# %%
# 7.4 Process the input text
# (A) tokenization
# Define parameters for the following process
vocab_size = 5000
oov_token = '<OOV>'
max_length = 200
embedding_dim = 64

# %%
# (B) define Tokenizer object
tokenizer = keras.preprocessing.text.Tokenizer(
    num_words=vocab_size,
    split=" ",
    oov_token=oov_token
)
tokenizer.fit_on_texts(X_train)

# %%
word_index = tokenizer.word_index
print(dict(list(word_index.items())[0:10]))

# %%
# (C) Transform texts into tokens
X_train_tokens = tokenizer.texts_to_sequences(X_train)
X_test_tokens = tokenizer.texts_to_sequences(X_test)

# %%
# (D) perform padding
X_train_padded = keras.utils.pad_sequences(X_train_tokens, 
                                           maxlen=max_length, 
                                           padding="post",
                                           truncating="post")
X_test_padded = keras.utils.pad_sequences(X_test_tokens, 
                                           maxlen=max_length, 
                                           padding="post",
                                           truncating="post")

# %%
reverse_word_index = dict([(value,key)for (key,value) in word_index.items()])

def decode_token(tokens):
    return " ".join([reverse_word_index.get(i,"?") for i in tokens])

print(X_train[2])
print("---------------")
print(decode_token(X_test_padded[2]))

# %%
# 8. Model development
#(A) Create a sequential model, then start with embedding layer
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size,embedding_dim))
#(B) Build RNN model, using bidirectional layer
model.add(keras.layers.Bidirectional(keras.layers.LSTM(48)))
model.add(keras.layers.Dense(48,activation='relu'))
model.add(keras.layers.Dense(len(np.unique(labels)), activation='softmax'))
model.summary()

# %%
# 9. compile the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# %%
# 10. model training
max_epochs = 20
early_stopping = keras.callbacks.EarlyStopping(patience=3)
PATH = os.getcwd()
logpath = os.path.join(PATH,"tensorboard_log",datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = callbacks.TensorBoard(logpath)
history = model.fit(
    X_train_padded, y_train,
    validation_data=(X_test_padded, y_test),
    epochs=max_epochs,
    callbacks=[early_stopping,tb]
)

# %%
# 11. plot graph to display training result
#(A) plot the loss graph
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training Loss','Validation Loss'])
plt.show()

# %%
# (B) accuracy graph
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Training Accuracy','Validation Accuracy'])
plt.show()

# %%
# 12. Evaluate the model
y_pred = model.predict(X_test_padded)
y_pred_classes = np.argmax(y_pred, axis=1)

# Convert categorical labels back to original labels
y_test_original = label_encoder.inverse_transform(y_test)
y_pred_original = label_encoder.inverse_transform(y_pred_classes)

# Calculate and print the F1 score
f1 = f1_score(y_test_original, y_pred_original, average='weighted')
print("Weighted F1 Score:", f1)

# Print classification report for more detailed metrics
print("Classification Report:\n", classification_report(y_test_original, y_pred_original))

# %%
# 13. Save important component so we can deploy the model in other application
# path to save tokenizer in json format to saved_models folder
from tensorflow.keras.models import load_model
import json
save_model_folder = os.path.join(PATH,'saved_models')
os.makedirs(save_model_folder, exist_ok=True) # Create the folder

tokenizer_save_path = os.path.join(save_model_folder,'tokenizer.json')

# Convert tokenizer to dictionary
tokenizer_dict = tokenizer.get_config()

# save tokenizer as json
with open(tokenizer_save_path, 'w') as json_file:
    json.dump(tokenizer_dict, json_file)

# %%
# Save model
model.save(os.path.join('saved_models','model.h5'))
# %%
# Get the model architecture
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model_architecture.png',show_shapes=True)
