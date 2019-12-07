# arabic-tweets-processor
an arabic tweets processor it built on the assumptions of gulf dialect as majority of stop words and other depends on this assumption.
Have Command Line Application that assumpe a column name ```text``` containes the tweets to be processed and cleaned for model building.
and also utils can be used as a stand-alone functions. all its inputes asssumes a type of Pandas Series. 



### Table of Contents

1. [Installation & dependencies](#Installation & dependencies)
2. [Commaned Line Application](#Command Line Application)


## Installation & dependencies <a name="Installation & dependencies"></a>

- Anaconda disrtbution with python vesion of 3^. should includes the necessary packages.
- emoji
- nltk


## Command Line Application

- Cleans and return 3 csv files. on a data set with text column containing tweets. **clean.py** 
  - Basic Usage : ```python clean.py data_path```<br/>
  - Options:
    - Set the mode : ```python clean.py data_path --mode mode_of_process``` for now only clean is implemented.
    - Set filename: ```python clean.py data_path --filename filename_of_extracted_feautres```
    - set how to handle emojies for cleaning: ```python clean.py data_path --handle_emojies remove_or_emoticion_or_emoji```
  - Output: A ```3 csv files``` 
  - 1. containes the cleand version of ```text``` 
  - 2. containes the term frequency of each word in tweets.
  - 3. containes the original tweets with extracted features.
