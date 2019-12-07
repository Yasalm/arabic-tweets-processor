import argparse
import utils
import pandas as pd
import dataProcessor
import time
import os

parser = argparse.ArgumentParser(prog='cleaner',
                                 description="Parser of cleaning script")

parser.add_argument(
    '--data_path', help='Provide Full path of data.', type=str)
parser.add_argument(
    '--mode', help='Clean, Classify,. Default=Clean.', type=str, default='clean')
parser.add_argument(
    '--filename', help="file name of cleaned data. Don't include .csv. default= extracted", type=str, default='extracted')
parser.add_argument(
    '--handle_emojies', help='How to handle emojies. [remove] to remove emojies. [emoticon] to keep emoticon. [emoji] to keep emojies، default=[emoticon]', type=str, default='emoticon')


args = parser.parse_args()

data_path = args.data_path
mode = args.mode
handle_emojies = args.handle_emojies
filename = args.filename

if not os.path.isfile(data_path):
    raise ValueError(f'file {data_path} not found')


def clean_data():

    start_time = time.time()

    processer = dataProcessor.DataProcessor()


    data = pd.read_csv(data_path, header=0)

    if mode.lower() == 'clean':

        data['word_count'] = utils.count_word(data.text)

        data['count_number'] = utils.count_numbers(data.text)

        data['emojies'] = utils.view_emojie(data.text)

        data['emoticons'] = utils.view_emoticon(data.text)

        data['len_tweet'] = utils.len_tweet(data.text)

        data['avg_words_len'] = utils.avg_word_len(data.text)

        data['count_stopwords'] = utils.count_stopwords(data.text)

        data['count_tagging'] = utils.count_tagging(data.text)

        data['flagged'] = utils.repeated_char(data.text)

        # data_copy.append([word_count, count_number, emojies, len_tweet, avg_words_len, count_stopwords, count_tagging], ignore_index=True)
        data.to_csv(filename+'.csv', index=False)

        tf = utils.term_freq(data.text)

        tf.to_csv('term_frequency.csv', index=False)


        data_pro, _ = processer.proccess_data(data.text, handle_emojies=handle_emojies)

        data_pro = pd.DataFrame(data_pro, columns=['text'])
        data_pro.append(data['label'])
        data_pro.to_csv('cleaned.csv', index=False)

        elapsed_time = time.time() - start_time
        print(f'Finished in {elapsed_time}')


    return None


if __name__ == "__main__":
    clean_data()
