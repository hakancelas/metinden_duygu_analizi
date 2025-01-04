# Gerekli kütüphaneleri import et
import nltk
from sklearn.model_selection import train_test_split  # Veri setini eğitim ve test için böler
from sklearn.feature_extraction.text import CountVectorizer  # Metinleri sayısal verilere dönüştürür
from sklearn.naive_bayes import MultinomialNB  # Naive Bayes sınıflandırıcı modelini kullanır
from sklearn.metrics import accuracy_score  # Modelin doğruluğunu ölçer

# NLTK kütüphanesindeki stopwords'ü indir
nltk.download('stopwords')  # Stopwords (bağlaçlar, edatlar vb.) İngilizce metinlerden çıkarılır

# Örnek metin verileri (etiketler: 1 = pozitif, 0 = negatif)
veriler = [
    ("I love this product", 1),  # Pozitif duygu örneği
    ("This is an amazing movie", 1),  # Pozitif duygu örneği
    ("I feel great about this experience", 1),  # Pozitif duygu örneği
    ("Horrible, I hate it", 0),  # Negatif duygu örneği
    ("This is the worst thing I've ever seen", 0),  # Negatif duygu örneği
    ("I would not recommend this to anyone", 0),  # Negatif duygu örneği
    ("Absolutely fantastic, I will buy again", 1),  # Pozitif duygu örneği
    ("I can't stand this", 0)  # Negatif duygu örneği
]

# Veriyi pandas DataFrame'e dönüştür
import pandas as pd  # Pandas, verileri düzenlemek için kullanılır
df = pd.DataFrame(veriler, columns=['Metin', 'Duygu'])  # 'Metin' ve 'Duygu' başlıklarıyla veri oluşturulur

# Veriyi eğitim ve test olarak ayır
X = df['Metin']  # Metin verisi
y = df['Duygu']  # Duygu etiketleri (0 ya da 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)  # Veri setini eğitim ve test olarak böler

# Metinleri sayısal verilere dönüştürmek için CountVectorizer kullan
vectorizer = CountVectorizer(stop_words=nltk.corpus.stopwords.words('english'))  # Stopwords çıkarılır
X_train_vec = vectorizer.fit_transform(X_train)  # Eğitim verisini sayısal verilere dönüştür
X_test_vec = vectorizer.transform(X_test)  # Test verisini sayısal verilere dönüştür

# Naive Bayes sınıflandırıcısını kullanarak modeli eğit
model = MultinomialNB()  # Naive Bayes modelini oluştur
model.fit(X_train_vec, y_train)  # Modeli eğitim verisi ile eğit

# Modeli test et ve doğruluğu hesapla
y_pred = model.predict(X_test_vec)  # Test verisi üzerinde tahmin yap
print("Doğruluk Skoru:", accuracy_score(y_test, y_pred))  # Modelin doğruluğunu yazdır

# Kullanıcıdan metin al ve tahmin yap
while True:
    kullanici_input = input("Bir metin girin (Çıkmak için 'q' tuşuna basın): ")  # Kullanıcıdan bir metin al
    if kullanici_input.lower() == 'q':  # Eğer kullanıcı 'q' tuşuna basarsa program sonlanır
        break
    kullanici_input_vec = vectorizer.transform([kullanici_input])  # Kullanıcının girdiği metni sayısal verilere dönüştür
    tahmin = model.predict(kullanici_input_vec)  # Kullanıcı metni üzerinde tahmin yap
    if tahmin[0] == 1:  # Eğer tahmin pozitifse
        print("Pozitif Duygu")  # Pozitif duygu olduğunu belirt
    else:  # Eğer tahmin negatifse
        print("Negatif Duygu")  # Negatif duygu olduğunu belirt