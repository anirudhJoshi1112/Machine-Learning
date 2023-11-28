import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
from tkinter import *
from tkinter import scrolledtext

# Download stopwords
nltk.download('stopwords')

# Read the dataset
news_dataset = pd.read_csv('/content/train.csv')

# Handle missing values
news_dataset = news_dataset.fillna('')

# Create a new column 'content'
news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']

# Perform text preprocessing
port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

news_dataset['content'] = news_dataset['content'].apply(stemming)

# Split into features and target
X = news_dataset['content'].values
Y = news_dataset['label'].values

# Vectorize the data
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)

# Split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Build and train the model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Create a basic GUI
def predict_news():
    text_to_predict = entry.get("1.0",'end-1c')
    X_new = vectorizer.transform([text_to_predict])
    prediction = model.predict(X_new)

    result_text.config(state=NORMAL)
    if prediction[0] == 0:
        result_text.delete(1.0, END)
        result_text.insert(END, 'The news is Real')
    else:
        result_text.delete(1.0, END)
        result_text.insert(END, 'The news is Fake')
    result_text.config(state=DISABLED)

# Main GUI window
window = Tk()
window.title("Fake News Detection")

# Input Textbox
label = Label(window, text="Enter News Text:")
label.pack()

entry = scrolledtext.ScrolledText(window, width=40, height=10, wrap=WORD)
entry.pack()

# Button to make prediction
button = Button(window, text="Predict", command=predict_news)
button.pack()

# Result Display
result_text = scrolledtext.ScrolledText(window, width=40, height=2, wrap=WORD, state=DISABLED)
result_text.pack()

# Run the GUI
window.mainloop()
