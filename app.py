import streamlit as st
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Load model dan vectorizer
model = joblib.load('svm_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Inisialisasi Stemmer Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Download resource NLTK jika belum tersedia
nltk.download('punkt')
nltk.download('stopwords')

# Daftar stopwords bahasa Indonesia
stop_words = set(stopwords.words('indonesian'))

# Tambahan stopwords kustom
additional_stopwords = {'nya', 'lah', 'sih', 'kan', 'kalo', 'gak', 'coba',
                        'sekarang', 'se', 'ini', 'ya', 'masa', 'mesti', 'emang',
                        'kalo', 'mau', 'dari', 'tapi', 'di', 'kalau', 'ngga', 'setiap',
                        'sampe', 'saya', 'cukup', 'untuk', 'tanpa', 'ada', 'terlalu',
                        'banyak', 'yang', 'dengan'}
stop_words.update(additional_stopwords)

# Fungsi untuk preprocessing teks
def clean_text(text):
    # Hapus URL
    text = re.sub(r'http\S+|www\S+', '', text)
    # Hapus mention dan hashtag
    text = re.sub(r'@\w+|#\w+', '', text)
    # Hapus emoji
    emoji_pattern = re.compile(
        "[" 
        u"\U0001F600-\U0001F64F"  # Emotikon wajah
        u"\U0001F300-\U0001F5FF"  # Simbol dan ikon lainnya
        u"\U0001F680-\U0001F6FF"  # Transportasi & simbol
        u"\U0001F1E0-\U0001F1FF"  # Bendera
        u"\U00002700-\U000027BF"  # Simbol tambahan
        u"\U000024C2-\U0001F251"  # Simbol lainnya
        "]+", flags=re.UNICODE
    )
    text = emoji_pattern.sub(r'', text)
    # Hapus tanda baca dan angka
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Case folding (huruf kecil)
    text = text.lower()
    # Tokenizing
    tokens = word_tokenize(text)
    # Stopword removal
    tokens = [word for word in tokens if word not in stop_words]
    # Stemming
    tokens = [stemmer.stem(word) for word in tokens]
    # Gabungkan kembali token menjadi satu string
    return ' '.join(tokens)

# Streamlit UI
st.title("Sentiment Analysis App")
st.subheader("Masukkan teks untuk dianalisis")

user_input = st.text_area("Teks Ulasan")

if st.button("Analisis Sentimen"):
    if user_input:
        cleaned_text = clean_text(user_input)  # Preprocessing
        transformed_text = vectorizer.transform([cleaned_text])  # Transformasi dengan vectorizer
        prediction = model.predict(transformed_text)[0]

        sentiment = "Positif" if prediction == 1 else "Negatif"
        st.write(f"**Hasil Sentimen:** {sentiment}")
    else:
        st.warning("Masukkan teks terlebih dahulu!")
