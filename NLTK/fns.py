# Initial imports
from calendar import day_abbr
import nltk
import pandas as pd
from pathlib import Path
from nltk.sentiment.vader import SentimentIntensityAnalyzer


from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.util import ngrams
from string import punctuation
import re
from collections import Counter

import pandasPalmer as pp
from datetime import datetime
from inspect import signature

#import NLTK.fns as nl

#%matplotlib inline

# Download/Update the VADER Lexicon
nltk.download("vader_lexicon")

def get_sentiment(score):
    """
    Calculates the sentiment based on the compound score.
    """
    result = 0  # Neutral by default
    if score >= 0.05:  # Positive
        result = 1
    elif score <= -0.05:  # Negative
        result = -1

    return result

# Function to create a dataframe from NewsApi articles
def Create_News_df(news_articles, language):
    """Creates a dataframe from NewsApi articles

    Args:
        news_articles (pd.DataFrame): Dictionary of articles from NewsApi
        language (str): language type e.g. en,fr etc.

    Returns:
        pd.DataFrame: Containing the author, title, description, date and text of the articles
    """
    articles_list = []
    for article in news_articles:
        try:
            author      = article["author"]
            title       = article["title"]
            description = article["description"]
            text        = article["content"]
            date_str    = article["publishedAt"][:10]

            articles_list.append({
                "author": author,
                "title": title,
                "description": description,
                "text": text,
                "date": datetime.fromisoformat(date_str),
                "language": language
            })
        except AttributeError:
            pass

    return pd.DataFrame(articles_list)

# Instantiate the lemmatizer
lemmatizer = WordNetLemmatizer()
stemp = PorterStemmer()

# Create a list of stopwords
sw = set(stopwords.words('english'))

# Tokenizer with various strategies
# Separate the word using a bar instead of a comma
bar_sep = lambda x: "|".join(x)

# Separate the words using as parameter
big_string = lambda x,sep=" ": sep.join(x)

# Generates ngrams based on the Parameters passed bigrams by default
bi_grams = lambda x : dict(Counter(ngrams(x, n=2)).most_common())
tri_grams = lambda x : dict(Counter(ngrams(x, n=3)).most_common())
N_grams   = lambda text, n=2: dict(Counter(ngrams(Regular_Tokenizer(text), n=n)).most_common())

# returns the words along with their frequency
word_count = lambda x: (dict(Counter(x).most_common(n=None)))

def tokenizer(text,Return_string=False, Post_Processor=None, Stem=False):
    """
    Takes text and turns them in to either a list of words, 
    a comma separated string of words or a list Dataframe listing the most common words

    Args:
        text (str):  text sentences or article.
        Return_string (bool, optional): Determines if a string or a list of string should be returned. Defaults to False.
        Post_Processor ([type], optional): The post processing function to perform on the results. Defaults to None.
        Stem (bool, optional): Performs an aditional step of steming the words
    Returns:
        list | str: Depends on the post processor passed into the function
    """

    # Remove the punctuation from text
    # https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string
    text = text.translate(str.maketrans('', '', punctuation))

    # Create a tokenized list of the words
    words = word_tokenize(text)    

    # Lemmatize words into root words
    lems = [lemmatizer.lemmatize(word) for word in words]

    if Stem:
        #  ****** Stemmer PorterStemmer ******
        lems = [stemp.stem(word=word) for word in lems]

    # Convert the words to lowercase
    output = [word.lower() for word in lems]

    # Remove the stop words
    tokens = [word for word in output if word not in sw]

    if Post_Processor is not None and callable(Post_Processor):
        sig = signature(Post_Processor)
        if sig.parameters.count == 1:
            return Post_Processor(tokens)

    if Return_string:
        return ",".join(tokens)
    else:
        return tokens

#_________________________________________________________________________
# Create Dataframe of N_grams
def Create_Bi_grams_df(corpus:str):
    #corpos = " ".join(bitcoin_news_df.text_tokens).replace(',',' ')
    bi_grams_result = tokenizer(corpus, Post_Processor=bi_grams)
    return pd.DataFrame(list(bi_grams_result.items()), columns=['word','count'])
def Create_Tri_grams_df(corpus:str):
    #corpos = " ".join(bitcoin_news_df.text_tokens).replace(',',' ')
    bi_grams_result = tokenizer(corpus, Post_Processor=tri_grams)
    return pd.DataFrame(list(bi_grams_result.items()), columns=['word','count'])
def Regular_Tokenizer(text):
    return tokenizer(text=text, Return_string=False, Post_Processor=None, Stem=False)

# Calculate the VADER Sentiment Score for Text columns in a DataFrame or those specified in a list
def Attach_Sentiment_Scores_2_df(df:pd.DataFrame, txt_cols=None):
    """Calculates the overall sentiment of text columns in a DataFrame and adds columns to the DataFrame with the results"""
    # Initialize the VADER sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Get the list of columns to work with
    if txt_cols is not None:
        if type(txt_cols) == list:
            pass
        elif type(txt_cols) == str:
            txt_cols = [txt_cols]
    else:   # There was no list of columns passed
        txt_cols = pp.Get_Columns_With_Text(df)
        if txt_cols is None:
            print("THERE WAS NO TEXT COLUMNS TO ANALYZE !!!!!!!!!!!!!!!!!!")
            return df

    # Get the sentiment for each row in the dataframe
    for col in txt_cols:
        # Create Sentiment Scoring Dictionary
        sentiment_dict = {
            f"{col}_compound": [],
            f"{col}_pos": [],
            f"{col}_neu": [],
            f"{col}_neg": [],
            f"{col}_sent": []
        }


        for _, row in df.iterrows():
            # Calculates the sentiment for the column
            sentiment = analyzer.polarity_scores(row[col])

            try:
                sentiment_dict[f"{col}_compound"].append(sentiment["compound"])
                sentiment_dict[f"{col}_pos"].append(sentiment["pos"])
                sentiment_dict[f"{col}_neu"].append(sentiment["neu"])
                sentiment_dict[f"{col}_neg"].append(sentiment["neg"])
                sentiment_dict[f"{col}_sent"].append(get_sentiment(sentiment["compound"]))
            except AttributeError as aexn:
                pass
        df = df.join(pd.DataFrame(sentiment_dict))

    return df


if __name__ == "__main__":
    print("Running the Personal NLTK Module")


    print(bar_sep("Testing Testing One two Three"))
    print(dict(Counter(tokenizer("Testing Testing One two Three")).most_common(10)))
    print(tokenizer("Testing Testing One two Three",Post_Processor=word_count))
    print(tokenizer("Testing Testing One two Three",Post_Processor=bar_sep))
    print(tokenizer("Testing Testing One two Three",Post_Processor=bi_grams))

    df = pd.DataFrame(
        data={
            'Col1':[1,2,3,4]
            ,'Col2':['A','B','C','D']
            ,'Col3':['asdf','asdf','zcsd','steve']
        }
    )
    #display(df)
    print(df)
    print(Attach_Sentiment_Scores_2_df(df,txt_cols="Col3"))