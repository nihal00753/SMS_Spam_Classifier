import pandas as pd
import pickle
import re
import string
import scipy.sparse as sp
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

ps = PorterStemmer()
sw = set(stopwords.words('english'))


def transform(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', 'url', text)
    text = re.sub(r'\b\d{7,}\b', 'phonenumber', text)
    text = re.sub(r'[£$€]\d+', 'currency', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return ' '.join([ps.stem(w) for w in text.split() if w not in sw])


df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['target', 'text']
df.drop_duplicates(inplace=True)
df['target'] = df['target'].map({'ham': 0, 'spam': 1})
df['clean'] = df['text'].apply(transform)

X_train, X_test, y_train, y_test = train_test_split(
    df['clean'], df['target'],
    test_size=0.2, random_state=2, stratify=df['target']
)

wv = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
cv = TfidfVectorizer(analyzer='char_wb', max_features=2000, ngram_range=(2, 4))

X_tr = sp.hstack([wv.fit_transform(X_train), cv.fit_transform(X_train)])
X_te = sp.hstack([wv.transform(X_test),      cv.transform(X_test)])

model = LogisticRegression(C=3, class_weight='balanced', max_iter=1000, random_state=42)
model.fit(X_tr, y_train)

y_pred = model.predict(X_te)
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")

pickle.dump({'word': wv, 'char': cv}, open('vectorizer.pkl', 'wb'))
pickle.dump(model, open('model.pkl', 'wb'))
print("Done. Run: streamlit run app.py")
