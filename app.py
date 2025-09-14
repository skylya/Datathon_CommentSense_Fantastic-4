import streamlit as st
import pandas as pd
import re
from textblob import TextBlob
from langdetect import detect, DetectorFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# ========== Load Dataset ==========
@st.cache_data
def load_data():
    return pd.read_csv("CombinedComments.csv")

data = load_data()

st.title("CommentSense - AI Comment Analysis App")
st.write("Analyze comments: language, spam detection, sentiment, categorization, and ML predictions.")

st.header("Dataset Preview")
if st.checkbox("Show raw data"):
    st.write(data.head())

DetectorFactory.seed = 0

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

# New column for language
data["language"] = data["textOriginal"].apply(detect_language)

print("Language distribution:")
print(data["language"].value_counts().head(10))  # see top 10 detected languages

# Filter for English only 
data = data[data["language"] == "en"]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'@\w+', '', text)
    text = text.replace("#", "")
    text = re.sub(r'[^a-zA-Z0-9\s\U0001F600-\U0001F64F]', '', text)
    text = re.sub(r'([!?.,])\1{2,}', r'\1', text)
    text = re.sub(r"\b(lol|omg|idk|lmao|haha|hehe)\b", "", text)
    return text.strip()

data["cleaned_comment"] = data["textOriginal"].apply(clean_text)

st.header("Language & Cleaning")
sample_text = st.text_input("Enter a comment to test language + cleaning:", "I love this product!!! 😍😍😍")
if sample_text:
    st.write("Detected Language:", detect_language(sample_text))
    st.write("Cleaned Text:", clean_text(sample_text))

# Emoji helper
def emoji_sentiment(text):
    if any(e in text for e in ["❤","💕","😘","😊","😂","😍"]):
        return "positive"
    if any(e in text for e in ["😡","😠","💔","😢"]):
        return "negative"
    return None

def get_sentiment(text):
    # Check emoji sentiment first
    emoji_score = emoji_sentiment(text)
    if emoji_score:
        return emoji_score

    # Fallback to TextBlob
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "positive"
    elif analysis.sentiment.polarity < 0:
        return "negative"
    else:
        return "neutral"
    
# Create sentiment column
data["sentiment"] = data["cleaned_comment"].apply(get_sentiment)

st.header("Sentiment Analysis")
user_text = st.text_area("Enter a comment:")
if user_text:
    st.write("Predicted Sentiment:", get_sentiment(user_text))

def is_spam(text):
    text = str(text).lower().strip()
    
    # 1. Emoji handling
    emoji_list = "❤💕😘😊😂😍😡😠💔😢😮😱🥰🔥✨💯👍🙏"
    emoji_count = sum(1 for ch in text if ch in emoji_list)

    # Emoji-only (spammy)
    if emoji_count > 5 and len(text.split()) == 0:
        return True

    # Emojis dominate (>60% of content is emojis)
    if emoji_count > 0 and (emoji_count / max(1, len(text))) > 0.6:
        return True

    # 2. Suspicious links or promo
    if "http" in text or "www" in text or "bit.ly" in text or "t.me" in text:
        return True

    # 3. Very short meaningless comment (only 1-2 chars)
    if len(text.split()) == 1 and len(text) < 3:
        return True

    # 4. Nonsense/random short alphanumeric mix
    if re.fullmatch(r"[a-zA-Z]*[0-9]+[a-zA-Z]*", text) and len(text) <= 6:
        return True

    # 5. Spammy phrases (hard rule)
    spammy_phrases = [
        "follow me", "check my channel", "subscribe", "buy now", 
        "click here", "dm me", "promo code", "giveaway", "visit my page"
    ]
    if any(phrase in text for phrase in spammy_phrases):
        return True
    
    # 6. Numbers + symbols together (common spam pattern)
    if re.search(r"[0-9]+.*[$%&]+|[$%&]+.*[0-9]+", text):
        return True
    
    # 7. Excessive punctuation or symbols
    if re.search(r"[!?.]{3,}", text) or re.search(r"[$]{3,}", text):
        return True
    
    # 8. Foreign/unicode junk (but allow common emojis handled earlier)
    if re.search(r"[^\x00-\x7F]+", text):
        return True

    return False

# Apply spam detection
data["is_spam"] = data["cleaned_comment"].apply(is_spam)

st.header("Spam Detection")
if user_text:
    st.write("Spam?", is_spam(user_text))

phrase_keywords = {
    "skincare": ["face mask", "face wash", "vitamin c", "anti aging", "anti-aging", "sheet mask", "eye cream", "glow up"],
    "makeup":  ["eye shadow", "lip gloss", "beauty blender", "beautyblender", "eye brow pencil", "foundation shade"],
    "fragrance": ["eau de parfum", "eau de toilette", "top notes", "base notes"],
    "service": ["shipping", "delivery", "refund", "customer service", "order arrived", "tracking number"],
    "price": ["expensive", "cheap", "price", "cost", "discount", "sale", "promo"],
    "question": ["how to use", "how do i", "where can i", "what is", "which shade", "how much"],
    "praise": ["i love", "so good", "so beautiful", "amazing", "best product", "highly recommend"],
    "hair": ["hair care", "hair style", "hairstyle", "hair cut", "hair color", "dandruff"]
}

word_keywords = {
    "skincare": [
        "cream","moisturizer","skin","lotion","sunscreen","serum","toner","cleanser",
        "mask","acne","hydrating","oil","exfoliator","spf","retinol","moisturiser", "natural"
    ],
    "makeup": [
        "lipstick","foundation","eyeliner","mascara","blush","makeup", "make up", "concealer","primer",
        "powder","highlighter","brow","palette","bronzer","contour","lashes","lipgloss","eyelash","eyeshadow","shade"
    ],
    "fragrance": [
        "perfume","scent","fragrance","cologne","parfum","spray","aroma","notes","scented"
    ],
    "service": [
        "shipping","delivery","refund","return","order","tracking","customer","support","warehouse","arrived","late"
    ],
    "price": [
        "price","expensive","cheap","discount","sale","deal","cost","worth"
    ],
    "question": [
        "how","what","where","which","why","help","does","is","can"  
    ],
    "praise": [
        "love","amazing","beautiful","best","nice","wow","loveit","cute","perfect","recommend","thanks","thankyou", "gorgeous", "great", "cool", "pretty", "slay", "fabulous"
    ],
    "hair": [
        "hair care", "hair style", "hairstyle", "hair cut", "haircuts", "hair color", "dandruff", "hair", "scalp", "wig"
    ]
}

def categorize_comment(text):
    text = str(text).lower()

    # 1. Check phrase keywords (multi-word)
    for cat, phrases in phrase_keywords.items():
        for phrase in phrases:
            if phrase in text:
                return cat

    # 2. Check word keywords (single words)
    for cat, words in word_keywords.items():
        for word in words:
            if word in text.split():  # ensures we match whole words
                return cat

    return "other"

st.header("Comment Categorization")
if user_text:
    st.write("Predicted Category:", categorize_comment(user_text))

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Simplified training
X = data["textOriginal"].astype(str).apply(clean_text)
y = data["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_tfidf, y_train)

st.header("ML Model Prediction (Sentiment)")
if user_text:
    X_new = vectorizer.transform([clean_text(user_text)])
    pred = clf.predict(X_new)[0]
    st.write("ML Predicted Sentiment:", pred)
    
# ==============================
# KPI Summary Report (Streamlit)
# ==============================
st.header("KPI Summary Report")

# 1. Spam vs Quality
spam_dist = data["is_spam"].value_counts(normalize=True) * 100
st.subheader("1. Spam vs Quality (%)")
st.bar_chart(spam_dist)

# 2. Distribution by Category
category_dist = data["category"].value_counts(normalize=True) * 100
st.subheader("2. Comment Distribution by Category (%)")
st.bar_chart(category_dist)

# 3. Sentiment Breakdown (overall)
sentiment_dist = data["sentiment"].value_counts(normalize=True) * 100
st.subheader("3. Overall Sentiment Distribution (%)")
st.bar_chart(sentiment_dist)

# 4. Sentiment within each Category
sentiment_per_cat = pd.crosstab(
    data["category"], 
    data["sentiment"], 
    normalize="index"
) * 100
st.subheader("4. Sentiment Breakdown per Category (%)")
st.dataframe(sentiment_per_cat.round(2))

