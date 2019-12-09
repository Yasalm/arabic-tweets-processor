import re
import pandas as pd
import numpy as np
import string
import emoji
import unicodedata as ud
from nltk.corpus import stopwords
import os





ar_stp = pd.read_fwf('stop_words.txt', header=None)
stop_words = set(stopwords.words('arabic') + list(ar_stp[0]))

def count_word(arr):
    """ takes an array of tweets and return the number of words per tweet
    Arguments:
    arr: Series. array of tweets.

    Returns:
    Series contains the count of words in each tweet
    """

    new_arr = []

    for text in arr:
        text = " ".join(word.strip() for word in re.split('#|_', text))
        text = text.replace('،', ' ')
        words = re.findall(r'[\u0600-\u06FF]+', text)

        new_arr.append(len([word for word in words if word not in stop_words]))
    # count = lambda x: len(str(x).split(" "))

    return pd.Series(new_arr)

def len_tweet(arr):
    """ takes an array of tweets and return the length of tweet per tweet including characters
    Arguments:
    arr: Series. array of tweets.

    Returns:
    Series contains the length of each tweet
    """
    return arr.str.len()

def avg_word_len(arr):
    """
    takes an array of tweets and return the the average length of words per tweet
    Arguments:
    arr: Series. array of tweets.

    Returns:
    a Series of the average of words length per tweet
    """

    # split = lambda x: x.split() # return a list of words. sep=" "

    new_arr = []

    # for text in arr:
    #     words = split(text)

    #     total_words_sum = (sum(len(word) for word in words)) # sum of the lenght of words in a tweet

    #     new_arr.append(total_words_sum/len(words)) # calculate the average

    arr_text = [" ".join(word.strip() for word in re.split('#|_', text)) for text in arr]
    arr_text = [text.replace('،', ' ') for text in arr_text]

    arr_text = [_remove_punctuation(text) for text in arr_text]

    arr_list_of_words  = [re.findall(r'[\u0600-\u06FF]+', text) for text in arr_text]

    arr_list_of_words = [_remove_stopwords(list) for list in arr_list_of_words]

    for list in arr_list_of_words:
        n = len(list)
        # print(n)
        total_words_sum = (sum(len(word) for word in list))
        # print(total_words_sum)
        new_arr.append(total_words_sum/n)

    return pd.Series(new_arr)

def count_stopwords(arr):
    """
    takes an array of tweets and return the the number of stopwords per tweet
    Arguments:
    arr: Series. array of tweets.

    Returns:
    a Series of the count of stopwords in each tweet
    """
    count = lambda x: len([x for x in x.split() if x in stop_words])

    return arr.apply(count)

def count_tagging(arr):
    """
    takes an array of tweets and return the the number of hashtags / mentions per tweet
    Arguments:
    arr: Series. array of tweets.

    Returns:
    a Series of the count of mentions and hashtags in each tweet
    """

    new_arr = []
    for text in arr:
        mentions = re.findall('@[^\s]+', text) # find mentions
        hashtags = re.findall(r'#([^\s]+)', text) # find hashtags

        # print(f'hashtags found {hashtags}, mentions found {mentions}')
        new_arr.append(len(mentions) + len(hashtags))
    return pd.Series(new_arr)


def count_numbers(arr):
    """
    takes an array of tweets and return the count of numbers present per tweet
    Arguments:
    arr: Series. array of tweets.

    Returns:
    a Series of the count of numbers presents per tweet
    """
    new_arr = []
    for text in arr:
        text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', text)
        numbes = re.findall(r'\d', text)
        new_arr.append(len(numbes))
    # count = lambda x: len([x for x in x.split() if x.isdigit()]) # count of digit present per tweet

    return pd.Series(new_arr)



def frequent_words(arr, topk=10, ascending=False):
    """
    takes an array of tweets and return the top [k] frequent words to all tweets
    Arguments:
    arr: Series. array of tweets.
    topk: int. top [k] words to return. default = 10.
    ascending: boolean. True: ascending, False: descending. default = False.

    Returns:
    a Series of the top [k] frequent words in tweets.
    """

    arr = get_arabic_words(arr, handle_emojies='remove')
    top_words = pd.Series(' '.join(arr).split()).value_counts(ascending=ascending)[:topk]
    return top_words

def view_emojie(arr):
    """
    takes an array of tweets and return the the emojies present in a tweet
    Arguments:
    arr: Series. array of tweets.

    Returns:
    a Series of the emojies of present per tweets.
    """

    new_arr = []
    for text in arr:
        # emojies = re.findall('@[^\s]+', tweet) # find emojies
        # print(f'emojies found {emojies})
        new_arr.append(_extract_emojis(text))
    return pd.Series(new_arr)

def _extract_emojis(str):
  return ''.join(c for c in str if c in emoji.UNICODE_EMOJI)

def view_emoticon(arr):
    """
    takes an array of tweets and return the the emoticon present in a tweet
    Arguments:
    arr: Series. array of tweets.

    Returns:
    a Series of the emoticon of present per tweets.
    """
    arr_emojies = [re.findall(emoji.get_emoji_regexp(), text) for text in arr]
    arr_emot = [_get_emoticon(item) for item in arr_emojies]

    new_arr = []
    for emoticon in arr_emot:
        new_arr.append(' '.join(emoticon))

    return pd.Series(new_arr)

def term_freq(arr):
    """
    takes an array of tweets and return the freqency of a word in a tweet
    Arguments:
    arr: Series. array of tweets.

    Returns:
    a dataframe of the frequency of words
    """
    arr = get_arabic_words(arr, handle_emojies='remove')

    df = arr.apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
    df.columns = ['words','tf']
    return df

def inverse_term_freq(arr):
    """
    takes an array of tweets and return the inverse freqency of a word in a tweet
    Arguments:
    arr: Series. array of tweets.

    Returns:
    a dataframe of the inverse frequency of words
    """
    n = arr.shape[0] # number of rows [tweets][documents]

    tf = term_freq(arr)
    for i, word in enumerate(tf['words']):
        sum_words_present = sum(arr.str.contains(word, regex=False))
        log = np.log(n/(sum_words_present + 1)) # +1 to avoid divison by zero.

        tf.loc[i, 'idf'] = log

    return tf

def tf_idf(arr):

    """
    takes an array of tweets and return the term frequency inverse document frequency (tf-idf) of a word in a tweet
    Arguments:
    arr: Series. array of tweets.

    Returns:
    a dataframe of the term frequency inverse document frequency (tf-idf) of words
    """

    tf = inverse_term_freq(arr)
    tf['tf-idf'] = tf['tf'] * tf['idf']
    return tf

def get_arabic_words(arr, handle_emojies='emoticon', remove_repeated_char=True):
    """
    the purpose of this function is to get arabic words only out of texts.
    takes an array of texts and return only arabic words
    Arguments:
    arr: Series. array of tweets.
    handle_emojies: String. emotiocon or keep or remove. How to handle emojies either to keep it as emoji or remove
    or keep the emoticon of an emoji
    remove_repeated_char: boolean. if True remove charactars that are repeated e.g ( احببب : احب), default True.
    Returns:
    a Series of arabic words per tweet
    """
    # remove (?, !, , ...)
    # punctuation = string.punctuation  + '، , ؟'
    # arr = [sentence.translate(str.maketrans('', '', punctuation)) for sentence  in arr]

    # keep only arabic words
    # print(f" : {arr}")


    if handle_emojies not in ['keep', 'remove', 'emoticon']:
        raise ValueError(f'Passed argument {handle_emojies} not a recognised argument.')
        
    arr_text = [" ".join(word for word in re.split('#|_', text)) for text in arr]
    arr_text = [text.replace('،', ' ') for text in arr_text]

    arr_text = [_remove_punctuation(text) for text in arr_text]

    arr_list_of_words  = [re.findall(r'[\u0600-\u06FF]+', text) for text in arr_text]

    arr_list_of_words = [_remove_stopwords(list) for list in arr_list_of_words]

    if remove_repeated_char:
        arr_list_of_words = [_handle_char(list) for list in arr_list_of_words]

    new_arr = []
    if handle_emojies.lower() == 'emoticon':
        arr_emojies = [re.findall(emoji.get_emoji_regexp(), text) for text in arr]
        arr_emot = [_get_emoticon(item) for item in arr_emojies]

        for text, emot in zip(arr_list_of_words, arr_emot):
            merged = text + emot # two lists..
            new_arr.append(' '.join(merged))

    if handle_emojies.lower() == 'keep':
        arr_emojies = [re.findall(emoji.get_emoji_regexp(), text) for text in arr]

        for text, emoj in zip(arr_list_of_words, arr_emojies):
            merged = text + emoj
            new_arr.append(' '.join(merged))

    if handle_emojies.lower() == 'remove':

        for text in arr_list_of_words:
            new_arr.append(' '.join(text))

    return pd.Series(new_arr)

def _remove_stopwords(arr):
    """"
    the purpose of this function is to remove stop words from a text

    Arguments:
    arr: List. list of words

    Returns:
    a List of words without stop words
    """
    return [word for word in arr if word not in stop_words]

def _handle_char(str):
    """
    takes a str of text represent a tweet and return text removed of repeated characters
    Arguments:
    arr: Series. array of tweets.

    Returns:
    a string removed of words removed of repeated characters ( e.g احببب: احب)
    """
    # print(f'non: {arr}')
    # print(f"OET: {list}")
    new_arr = []
    check_len = lambda x: len(x) >= 3
    for word in str:
        word = re.sub(r'(.)\1+', r'\1', word)
        if check_len(word):
            new_arr.append(word)
    # print(f'non: {arr}')
    # arr = [word for word in list if len(word) >= 3]
    return new_arr

def repeated_char(arr):
    """
    takes a arr of texts represent a tweet and return text removed of repeated characters
    Arguments:
    arr: Series. array of tweets.

    Returns:
    a string removed of words removed of repeated characters ( e.g احببب: احب)
    """

    arr_text = [" ".join(word.strip() for word in re.split('#|_', text)) for text in arr]
    arr_text = [text.replace('،', ' ') for text in arr_text]

    arr = [_remove_punctuation(text) for text in arr]
    arr = [re.sub(r'\d', " ", text) for text in arr]
    arr_list_of_words  = [re.findall(r'[\u0600-\u06FF]+', text) for text in arr]



    new_arr = []
    for list in arr_list_of_words:
        # words = [word for word in list if not word.isdigit()]
        words = [word for word in list if re.findall(r'(.)\1{2}', word)]
        new_arr.append(' '.join(words))
    return pd.Series(new_arr)

def _remove_punctuation(str):
    """
    takes a str of text represent a tweet and return text removed of punctuation
    Arguments:
    arr: Series. array of tweets.

    Returns:
    a string removed of punctuation
    """
    return ''.join(c for c in str if not ud.category(c).startswith('P'))

def _get_emoticon(arr):
    """"
    the purpose of this function is to get the matching emoticon of an emoji

    takes an array of emojies and return only emoticon per text
    Arguments:
    arr: List. List contains  List of emojies. where each list represnt an observation [row].

    Returns:
    a List of emotiocn per tweet
    """
    new_arr = []
    for item in arr:
        new_arr.append(emoji.demojize(item))
    return new_arr

# def remove_spam(arr):

#     """
#     takes a array of tweets and remove spam
#     Arguments:
#     arr: Series. array of tweets.

#     Returns:
#     a Series of tweets removed of spam tweets
#     """


#     return arr

def df_to_pdf(df, filename):
    """
    (Required dependencie: https://pypi.org/project/pdfkit/)

    takes a dataframe and create a pdf page
    Arguments:
    df: Pandas Dataframe. containes data to map to pdf page.
    filename: String. filename of a pdf page

    Returns:
    None
    """

    try:
        import pdfkit as pdf

        html = df.to_html()

        with open(filename+'.html', "w", encoding="utf-8") as file:

            file.writelines('<meta charset="UTF-8">\n')
            file.write(html)

        pdf.from_file(filename+'.html', filename+'.pdf')

        os.remove(filename+'.html')

    except:
       pass
