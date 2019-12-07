import nltk
from nltk.corpus import stopwords
from string import punctuation
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.stem.isri import ISRIStemmer
from nltk.tokenize.treebank import TreebankWordDetokenizer
import utils
import emoji


# sad_emoticons = r'^[[:-(", ":(", ":-|", ";-(", ";-<", "|-{"]
# happy_emoticons = {":-)", ":)", ":o)", ":-}", ";-}", ":->", ";-)"}
ar_stp = pd.read_fwf('stop_words.txt', header=None)


class DataProcessor:

    def __init__(self):
        # must-add : more stop words likes: كأنه,  اللي
        self._stopWords = set(stopwords.words('arabic') + list(punctuation) + list(ar_stp[0]))


    def proccess_data(self, arr, handle_emojies,  stem=False, tokinzie=False, threshold=0):
        """
        takes an array of string/texts and remove stopwords, punctuation, and return the root of the word.

        Arguments:
        arr: numpy array or list. texts to be cleaned.
        stem: boolean. True return the stme of the word, False keep the original.
        tokinzie: boolean. True return a tokinzie version of a text. False Return as text.
        threshold: int. set the threshold of lenght of text to keep.

        Returns:
        an array of cleaned text.
        """

        missing_values = arr.isna().sum()
        if missing_values > 0:
            raise ValueError(f'found {missing_values} missing values')

        # if threshold > 0:
        #     arr = [item for item in arr if len(arr) > threshold]

        # processed_arr = [self.ـprocess_text(item, stem) for item in arr]
        processed_arr = utils.get_arabic_words(arr, handle_emojies=handle_emojies)

        # for i, item in enumerate(processed_arr):
        #     print(len(item))

        if tokinzie:
            return processed_arr, arr

        # arr_detokenized = [TreebankWordDetokenizer().detokenize(word) for word in processed_arr]

        return processed_arr, arr



    # def ـprocess_text(self, value, stem):
    #     value = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', value) # remove URLs
    #     value = re.sub('@[^\s]+', '', value) # remove usernames
    #     # value = re.sub('[0-9]', '', value)
    #     value = re.sub('[a-zA-Z]', '', value)
    #     # value = re.sub(r'\d', '', value)
    #     value = re.sub('^[\u0621-\u064A\u0660-\u0669 ]+$', '', value) # remove special characters like ('\n')
    #     value = re.sub(r'#([^\s]+)', r'\1', value) # remove the # in #hashtag
    #     value = re.sub(r'(.)\1+', r'\1', value)

    #     value = re.sub(emoji.get_emoji_regexp(), r"", value)



    #     value = word_tokenize(value)


    #     if stem:
    #         value = self._stem(value)
    #     return [word for word in value if word not in self._stopWords]


    # def _stem(self, arr):

    #     _st = ISRIStemmer()

    #     value = [_st.pre32(word) for word in arr]
    #     value = [_st.suf32(word) for word in arr]
    #     # value = [_st.waw(word) for word in arr]
    #     value = [_st.stem(word) for word in arr]

    #     return value

