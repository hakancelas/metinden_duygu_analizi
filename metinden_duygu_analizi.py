# Gerekli kütüphaneleri import et
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# NLTK kütüphanesindeki stopwords'ü indir
nltk.download('stopwords')

# Örnek metin verileri (etiketler: 1 = pozitif, 0 = negatif)
veriler = [
    ("I love this product", 1),
    ("This is an amazing movie", 1),
    ("I feel great about this experience", 1),
    ("Horrible, I hate it", 0),
    ("This is the worst thing I've ever seen", 0),
    ("I would not recommend this to anyone", 0),
    ("Absolutely fantastic, I will buy again", 1),
    ("I can't stand this", 0)
]

# Veriyi pandas DataFrame'e dönüştür
import pandas as pd
df = pd.DataFrame(veriler, columns=['Metin', 'Duygu'])

# Veriyi eğitim ve test olarak ayır
X = df['Metin']
y = df['Duygu']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Metinleri sayısal verilere dönüştürmek için CountVectorizer kullan
vectorizer = CountVectorizer(stop_words=nltk.corpus.stopwords.words('english'))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Naive Bayes sınıflandırıcısını kullanarak modeli eğit
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Modeli test et ve doğruluğu hesapla
y_pred = model.predict(X_test_vec)
print("Doğruluk Skoru:", accuracy_score(y_test, y_pred))

# Kullanıcıdan metin al ve tahmin yap
while True:
    kullanici_input = input("Bir metin girin (Çıkmak için 'q' tuşuna basın): ")
    if kullanici_input.lower() == 'q':
        break
    kullanici_input_vec = vectorizer.transform([kullanici_input])
    tahmin = model.predict(kullanici_input_vec)
    if tahmin[0] == 1:
        print("Pozitif Duygu")
    else:
        print("Negatif Duygu")