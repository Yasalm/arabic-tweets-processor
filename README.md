# arabic-tweets-processor
An arabic tweets processor it built on the assumptions of gulf dialect as majority of stop words and other depends on this assumption.
A Command Line Application that assumpe a column name ```text``` containes the tweets to be processed and cleaned for model building.
utils could also be used as a stand-alone. As all its inputes asssumes a type of Pandas Series. 


## Installation & dependencies <a name="Installation & dependencies"></a>

- Anaconda disrtbution with python vesion of 3^. should includes the necessary packages.
- emoji
- nltk
- arabic_reshaper
- bidi.algorithm 
- wordcloud 


## Command Line Application <a name="Commaned Line Application"></a>

- Cleans and return 3 csv files. on a data set with text column containing tweets. **clean.py** 
  - Basic Usage : ```python clean.py --data_path data```<br/>
  - Options:
    - Set the mode : ```python clean.py --data_path data --mode mode_of_process``` for now only clean is implemented.
    - Set filename: ```python clean.py data_path --filename filename_of_extracted_feautres```
    - Set how to handle emojies for cleaning: ```python clean.py --data_path --handle_emojies remove_or_emoticion_or_emoji```
  - Output: A ```3 csv files``` 
    - containes the cleand version of ```text``` 
    - containes the term frequency of each word in provided tweets.
    - containes the original tweets with extracted features.
