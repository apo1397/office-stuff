import nltk
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import contractions
import emoji
from gensim.models.phrases import Phrases, Phraser
import logging
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Configure logging (adjust level as needed)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_instagram_comment(text):
    """
    Clean Instagram comment text for sentiment analysis.
    
    Steps:
    - Remove URLs.
    - Remove user mentions.
    - Convert hashtags by stripping the '#' symbol.
    - Convert emojis to their textual descriptions.
    - Normalize repeated punctuation and extra whitespace.
    """
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # Remove user mentions (e.g., @username)
    text = re.sub(r"@\w+", "", text)
    # Normalize hashtags (remove '#' but keep the text)
    text = re.sub(r"#(\w+)", r"\1", text)
    # Convert emojis to text (e.g., 😊 becomes :smiling_face:)
    text = emoji.demojize(text, delimiters=(" ", " "))
    # Normalize repeated punctuation (e.g., "!!!" -> "!")
    text = re.sub(r'([!?.])\1+', r'\1', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def count_words_emojis(text):
    """Counts words (ignoring emojis and punctuation) and emojis in text."""
    logging.debug(f"Input text: {text}")

    # Improved emoji detection regex (handles consecutive emojis)
    emoji_pattern = re.compile(r'(['
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               '])+', flags=re.UNICODE)

    emojis = emoji_pattern.findall(text)
    emoji_count = emoji.emoji_count(text)

    # Remove emojis from the text
    text_without_emojis = emoji_pattern.sub('', text)

    # Remove punctuation and extra whitespace
    text_cleaned = re.sub(r'[^\w\s]', '', text_without_emojis).strip()

    # Split into words
    words = text_cleaned.split()
    word_count = len(words)

    logging.debug(f"Word count: {word_count}, Emoji count: {emoji_count}")
    return word_count, emoji_count

def remove_stopwords(text):
    """Removes stop words from a text (handles contractions)."""
    logging.debug(f"Removing stop words from: {text}")
    expanded_text = expand_contractions(text)  # Expand contractions before tokenization
    words = word_tokenize(expanded_text)
    filtered_words = [word for word in words if word.lower() not in stop_words and word.isalnum()]
    logging.debug(f"Filtered words: {filtered_words}")
    return filtered_words

def expand_contractions(text):
    """Expands contractions in a text."""
    logging.debug(f"Expanding contractions in: {text}")
    expanded_text = contractions.fix(text)
    logging.debug(f"Expanded text: {expanded_text}")
    return expanded_text

def lemmatize_words(words):
    """Lemmatizes a list of words."""
    logging.debug(f"Lemmatizing words: {words}")
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    logging.debug(f"Lemmatized words: {lemmatized_words}")
    return lemmatized_words

def detect_phrases(sentence, words_filtered):
    # Load the small English model
    nlp = spacy.load('en_core_web_sm')
    # Process the sentence
    emoji_pattern = re.compile(r'(['
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               '])+', flags=re.UNICODE)

    words_no_emojis = [word for word in words_filtered if not emoji_pattern.fullmatch(word)]
    sentence = " ".join(words_no_emojis)
    doc = nlp(sentence)
    # Extract noun chunks as phrases
    phrases = [chunk.text for chunk in doc.noun_chunks]
    return phrases

def extract_handles_hashtags(text):
    """Extracts handles and hashtags from text."""
    logging.debug(f"Input text: {text}")
    handles = re.findall(r'@(\w+)', text)
    hashtags = re.findall(r'#(\w+)', text)
    logging.debug(f"Handles: {handles}, Hashtags: {hashtags}")
    return handles, hashtags

def detect_sentiment_and_emotion_instagram(text):
    """
    Detect sentiment and emotion from an Instagram comment.
    
    This function:
    - Preprocesses the comment to handle URLs, mentions, hashtags, emojis, and punctuation.
    - Uses VADER sentiment analysis to compute sentiment scores.
    - Determines an overall sentiment and primary emotion based on VADER's scores.
    
    Returns a dictionary containing the original text, cleaned text, overall sentiment,
    primary emotion (positive, neutral, or negative), and detailed sentiment scores.
    """
    logging.debug(f"Original text: {text}")
    
    # Clean the comment text
    cleaned_text = clean_instagram_comment(text)
    logging.debug(f"Cleaned text: {cleaned_text}")
    
    # Initialize the VADER sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(cleaned_text)
    logging.debug(f"Sentiment scores: {sentiment_scores}")

    # Determine overall sentiment based on compound score thresholds
    compound = sentiment_scores['compound']
    if compound >= 0.05:
        sentiment = 'Positive'
    elif compound <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    logging.debug(f"Overall sentiment: {sentiment}")
    
    # Determine the primary emotion from the positive, neutral, and negative scores
    emotions = {k: sentiment_scores[k] for k in ['pos', 'neu', 'neg']}
    primary_emotion = max(emotions, key=emotions.get)
    
    logging.debug(f"Primary emotion: {primary_emotion}")
    
    return {
        # 'original_text': text,
        # 'cleaned_text': cleaned_text,
        'sentiment': sentiment,
        'primary_emotion': primary_emotion,
        'sentiment_scores': sentiment_scores
    }     

def process_comment(comment, phrases_model=None):
    """Processes a single comment."""
    logging.debug(f"Processing comment: {comment}")
    try:
        handles, hashtags = extract_handles_hashtags(comment)
        words = word_tokenize(comment)
        word_count, emoji_count = count_words_emojis(comment)
        words_filtered = remove_stopwords(comment) #passing text instead of words
        expanded_comment = expand_contractions(comment)
        lemmatized_words = lemmatize_words(words_filtered)
        detected_phrases = detect_phrases(comment, words_filtered)  # Passing words_filtered
        sentiment_emotion = detect_sentiment_and_emotion_instagram(comment)

        return pd.DataFrame({
            'comment': [comment],
            'handles': [handles],
            'hashtags': [hashtags],
            'word_count': [word_count],
            'emoji_count': [emoji_count],
            'filtered_words': [words_filtered],
            'expanded_comment': [expanded_comment],
            'lemmatized_words': [lemmatized_words],
            'detected_phrases': [detected_phrases],
            'sentiment_emotion': [sentiment_emotion]
        })
    except Exception as e:
        logging.exception(f"Error processing comment: {e}") #Logs the full traceback
        return pd.DataFrame() # Return an empty DataFrame if error occurs


comments = [
    "Hey @user1, check out this awesome😃 #newproduct! 😃",
    "I don't like this product. #badquality",
    "Amazing work! @user2 and @user3, you guys rock! 💪",
    "This is another comment with some more words and emojis 🎉🥳.",
    "Just received my order! 📦 It looks fantastic! #happycustomer",
    "Can't believe how good this is! @brandname, you nailed it! 👏😍",
    "This outfit is so stunning! Perfect for the summer vibes! 🌞🌼 #fashion",
    "Love the aesthetics of this post! ✨✨ #photography #inspiration",
    "So delicious! 😋 Just tried your recipe and it was a hit! #foodie",
    "Where did you get those shoes? I need them in my life! 👟❤️",
    "This place is beautiful! Would love to visit someday! 🏖️ #travelgoals",
    "Your makeup looks flawless! Could you share the products? 💄✨",
    "What a stunning view! 😍🏞️ #nature #explore",
]

# Train phrase model if needed (requires a larger corpus for better results)
# sentences = [word_tokenize(comment) for comment in comments]  #Use a much larger corpus here
# phrases = Phrases(sentences, min_count=5, threshold=10) # Adjust min_count and threshold
# bigram = Phraser(phrases)


processed_comments = []
for comment in comments:
    processed_data = process_comment(comment) # , bigram) Uncomment if you train a phrase model
    processed_comments.append(processed_data)

df = pd.concat(processed_comments, ignore_index=True)
# Write the DataFrame to a CSV file
output_filename = 'processed_comments.csv'  #Choose your desired filename
df.to_csv(output_filename, index=False, encoding='utf-8') #index=False prevents writing row indices
print(f"Data written to '{output_filename}'")