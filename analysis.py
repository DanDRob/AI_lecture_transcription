import logging
import re
from collections import Counter

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.probability import FreqDist
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    # Create dummy objects/functions if nltk is not available to prevent import errors elsewhere
    # though the functions using them will raise an error if called.
    class FreqDist:
        pass
    def sent_tokenize(text): return [text]
    def word_tokenize(text): return text.split()
    stopwords = type('stopwords', (object,), {'words': lambda lang: []})()

try:
    from rake_nltk import Rake
    RAKE_AVAILABLE = True
except ImportError:
    RAKE_AVAILABLE = False
    class Rake:
        def extract_keywords_from_text(self, text): pass
        def get_ranked_phrases_with_scores(self): return []

logger = logging.getLogger(__name__)

# Ensure NLTK data is available (user might need to download them manually)
# This should ideally be handled by a setup script or first-run check in the main app.
_NLTK_DATA_DOWNLOADED = False
def ensure_nltk_data():
    global _NLTK_DATA_DOWNLOADED
    if not NLTK_AVAILABLE or _NLTK_DATA_DOWNLOADED:
        return
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        _NLTK_DATA_DOWNLOADED = True
        logger.info("NLTK 'punkt' and 'stopwords' resources found.")
    except nltk.downloader.DownloadError:
        logger.warning("NLTK 'punkt' and/or 'stopwords' not found. Attempting to download.")
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            _NLTK_DATA_DOWNLOADED = True
            logger.info("NLTK 'punkt' and 'stopwords' downloaded successfully.")
        except Exception as e:
            logger.error(f"Failed to download NLTK resources: {e}")
            logger.error("Please install them manually: run Python, then `import nltk; nltk.download('punkt'); nltk.download('stopwords')`")
    except AttributeError:
        # nltk might be a dummy object if import failed
        logger.error("NLTK is not properly installed or available.")


def extract_keywords(text, max_keywords=10):
    """Extracts keywords from text using RAKE (Rapid Automatic Keyword Extraction)."""
    if not RAKE_AVAILABLE:
        logger.error("RAKE-NLTK library is not installed. Cannot extract keywords.")
        return []
    if not NLTK_AVAILABLE:
        logger.error("NLTK library is not installed. RAKE depends on NLTK. Cannot extract keywords.")
        return []
    
    ensure_nltk_data()
    if not _NLTK_DATA_DOWNLOADED:
        logger.warning("NLTK data not available, keyword extraction might be suboptimal or fail.")

    if not text or not isinstance(text, str) or len(text.strip()) < 20: # Basic check for meaningful text
        logger.info("Text too short or invalid for keyword extraction.")
        return []

    try:
        r = Rake(
            stopwords=stopwords.words('english'), # Use NLTK's English stopwords
            punctuations=None, # Use default punctuations
            min_length=1, # Minimum words in a keyword
            max_length=3  # Maximum words in a keyword
        )
        r.extract_keywords_from_text(text)
        # Get ranked phrases with scores, take top N
        ranked_phrases_with_scores = r.get_ranked_phrases_with_scores()
        keywords = [phrase for score, phrase in ranked_phrases_with_scores[:max_keywords]]
        logger.info(f"Extracted keywords: {keywords}")
        return keywords
    except Exception as e:
        logger.error(f"Error during keyword extraction: {e}", exc_info=True)
        return []

def generate_extractive_summary(text, num_sentences=3, min_sentence_len=5):
    """
    Generates a simple extractive summary from text.
    Selects sentences with the highest frequency of significant words.
    """
    if not NLTK_AVAILABLE:
        logger.error("NLTK library is not installed. Cannot generate summary.")
        return "Summary generation requires NLTK."

    ensure_nltk_data()
    if not _NLTK_DATA_DOWNLOADED:
        logger.warning("NLTK data not available, summary generation might be suboptimal or fail.")
        return "Summary generation failed due to missing NLTK data."

    if not text or not isinstance(text, str) or len(text.strip()) < 50: # Need enough text for a summary
        logger.info("Text too short or invalid for summarization.")
        return "Text too short to summarize."

    try:
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            logger.info("Text has fewer or equal sentences than requested summary length. Returning original text.")
            return text # If text is already short, return as is

        # Tokenize words and remove stopwords and punctuation
        stop_words = set(stopwords.words('english'))
        # Normalize words: lowercase and keep only alphanumeric
        words = [re.sub(r'[^a-zA-Z0-9]', '', word.lower()) for word in word_tokenize(text) if word.isalnum() and word.lower() not in stop_words]
        
        if not words:
            logger.info("No significant words found after filtering. Cannot generate summary.")
            return "Could not find significant words to generate summary."

        word_frequencies = FreqDist(words)
        most_frequent_words = {word for word, freq in word_frequencies.most_common(10)} # Consider top 10 frequent as important

        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            words_in_sentence = [re.sub(r'[^a-zA-Z0-9]', '', word.lower()) for word in word_tokenize(sentence) if word.isalnum()]
            if len(words_in_sentence) < min_sentence_len: # Skip very short sentences
                continue
            
            score = 0
            for word in words_in_sentence:
                if word in most_frequent_words:
                    score += word_frequencies[word] # Score based on overall frequency of important words
            
            # Normalize by sentence length (alternative: don't normalize, or sqrt(length))
            if len(words_in_sentence) > 0:
                 # sentence_scores[i] = score / len(words_in_sentence)
                 sentence_scores[i] = score # Simpler: sum of frequencies of important words
            else:
                sentence_scores[i] = 0

        # Select top N sentences, preserving original order
        # Sort by score, then by original index to maintain some order for tied scores (though not strictly needed for extractive)
        ranked_sentence_indices = sorted(sentence_scores, key=lambda i: (-sentence_scores[i], i))
        
        selected_indices = sorted(ranked_sentence_indices[:num_sentences])
        summary_sentences = [sentences[i] for i in selected_indices]
        summary = " ".join(summary_sentences)
        
        logger.info(f"Generated summary: {summary}")
        return summary

    except Exception as e:
        logger.error(f"Error during summary generation: {e}", exc_info=True)
        return "Error generating summary."

if __name__ == '''__main__''':
    logging.basicConfig(level=logging.INFO)
    ensure_nltk_data() # Crucial for testing

    sample_text = (
        "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence "
        "concerned with the interactions between computers and human language, in particular how to program computers "
        "to process and analyze large amounts of natural language data. Challenges in natural language processing frequently "
        "involve speech recognition, natural language understanding, and natural language generation. The history of NLP "
        "generally started in the 1950s, although work can be found from earlier periods. In 1950, Alan Turing published "
        "an article titled \"Computing Machinery and Intelligence\" which proposed what is now called the Turing test as a "
        "criterion of intelligence. This criterion depends on the ability of a computer program to impersonate a human in a "
        "real-time written conversation with a human judge, sufficiently well that the judge is unable to distinguish reliably "
        "between the program and a real human."
    )
    sample_text_short = "This is a very short text. It cannot be summarized effectively but keywords might work."
    sample_text_empty = ""

    print("--- Keyword Extraction Test ---")
    keywords = extract_keywords(sample_text)
    print(f"Keywords from long text: {keywords}")
    keywords_short = extract_keywords(sample_text_short)
    print(f"Keywords from short text: {keywords_short}")
    keywords_empty = extract_keywords(sample_text_empty)
    print(f"Keywords from empty text: {keywords_empty}")

    print("\n--- Summarization Test ---")
    summary = generate_extractive_summary(sample_text, num_sentences=2)
    print(f"Summary from long text (2 sentences):\n{summary}")
    summary_short = generate_extractive_summary(sample_text_short)
    print(f"Summary from short text:\n{summary_short}")
    summary_empty = generate_extractive_summary(sample_text_empty)
    print(f"Summary from empty text:\n{summary_empty}")

    # Test with text that might have few unique important words
    repetitive_text = (
        "The cat sat on the mat. The cat was black. The mat was flat. The cat liked the mat. "
        "A dog ran by. The dog was brown. The cat watched the dog. The dog barked."
    )
    print("\n--- Repetitive Text Test ---")
    keywords_rep = extract_keywords(repetitive_text, max_keywords=5)
    print(f"Keywords from repetitive text: {keywords_rep}")
    summary_rep = generate_extractive_summary(repetitive_text, num_sentences=2)
    print(f"Summary from repetitive text (2 sentences):\n{summary_rep}")

    # Test NLTK not available scenario (manual mock needed if NLTK is actually installed)
    # NLTK_AVAILABLE = False # This would need to be set before module load or by clever patching
    # RAKE_AVAILABLE = False
    # print("\n--- NLTK Not Available Test (Simulated) ---")
    # if not NLTK_AVAILABLE:
    #     print(f"Keywords (NLTK unavailable): {extract_keywords(sample_text)}")
    #     print(f"Summary (NLTK unavailable): {generate_extractive_summary(sample_text)}")
    # else:
    #     print("Cannot simulate NLTK unavailable as it is installed.") 