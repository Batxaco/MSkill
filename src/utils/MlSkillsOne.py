import glob
import string
import sys
from difflib import SequenceMatcher

import locationtagger
import pandas as pd
import textacy
from matplotlib import pyplot as plt
from wordcloud import STOPWORDS, WordCloud


class TextAnalizer:

    def __init__(self):
        pass
        self.all_records_raw_df = self.read_data_individual_topics()
        self.all_records_cleaned_df = self.data_cleaner()
        self.target_phrase = "Jeu de Paume is an excellent art gallery in Paris"

    @staticmethod
    def read_data_individual_topics():

        path = '../data/individualTopics_27-01-22/'
        all_records = []

        for fname in glob.glob(path + '*.pickle'):
            obj = pd.read_pickle(fname)
            record = [obj['id'], obj['name'], obj['audience_size'],
                      obj['country'], obj['topic']]
            all_records = all_records + [record]

        if len(all_records) != 0:
            all_records_df = pd.DataFrame.from_records(all_records,
                                                       columns=['id', 'name', 'audience_size', 'country', 'topic'])
        else:
            sys.exit("File data not found")

        return all_records_df

    @staticmethod
    def number_of_verb(text):
        verbs = []
        pattern = [{'POS': 'VERB', 'OP': '?'},
                   {'POS': 'VERB', 'OP': '+'}]
        doc = textacy.make_spacy_doc(text, lang='en_core_web_sm')
        lists = textacy.extract.matches.token_matches(doc, [pattern])
        for list in lists:
            verbs.append(list.text)

        return len(verbs)

    @staticmethod
    def number_letters(text):
        return len([i for i in text if i.isalpha()])

    @staticmethod
    def location(text):
        place_entity = locationtagger.find_locations(text=text)
        countries = place_entity.countries
        regions = place_entity.regions
        cities = place_entity.cities
        X = countries + regions + cities
        return X

    @staticmethod
    def similarity(row):
        return SequenceMatcher(None, row, "Jeu de Paume is an excellent art gallery in Paris").ratio()

    def data_cleaner(self):

        self.all_records_raw_df["name_cleaned"] = self.all_records_raw_df.name \
            .apply(lambda row: row.translate(str.maketrans('', '', string.punctuation)))
        """
        self.all_records_raw_df["number_of_verb"] = self.all_records_raw_df.name_cleaned\
            .apply(lambda row: self.number_of_verb(row))

        self.all_records_raw_df["number_words"] = self.all_records_raw_df.name_cleaned\
            .apply(lambda row: len(row.split()))

        self.all_records_raw_df["number_letters"] = self.all_records_raw_df.name_cleaned\
            .apply(lambda row: self.number_letters(row))

        self.all_records_raw_df["number_letters_words_verbs"] = self.all_records_raw_df.number_letters.\
            apply(lambda row: [row]) + self.all_records_raw_df.number_words.apply(lambda row: [row]) + \
            self.all_records_raw_df.number_of_verb.apply(lambda row: [row])

        self.all_records_raw_df["name_entity"] =  self.all_records_raw_df.name_cleaned\
            .apply(lambda row: self.location(row))  
        """
        self.all_records_raw_df["similarity"] = self.all_records_raw_df.name_cleaned \
            .apply(lambda row: self.similarity(row))

        self.all_records_raw_df["target_phrase"] = "Jeu de Paume is an excellent art gallery in Paris"

        return self.all_records_raw_df

    def word_cloud(self):

        strings = self.all_records_raw_df.name_cleaned
        comment_words = ''
        stopwords = set(STOPWORDS)
        for val in strings:

            val = str(val)

            tokens = val.split()

            for i in range(len(tokens)):
                tokens[i] = tokens[i].lower()

            comment_words += " ".join(tokens) + " "

        wordcloud = WordCloud(width=800, height=800,
                              background_color='white',
                              stopwords=stopwords,
                              min_font_size=10).generate(comment_words)

        plt.figure(figsize=(8, 8), facecolor=None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)

        plt.show()
