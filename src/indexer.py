import heapq
import numpy as np
from collections import OrderedDict
from typing import *
from nltk.corpus import stopwords
from emoji import UNICODE_EMOJI
from string import punctuation
from nltk.tokenize import TweetTokenizer


class InvertedIndex(dict):
        """
        This class creates an inverted index of a .csv file
        that contains information and content from a number of tweets.

        This class inherits from the default dictionary class in Python.
        This was done primarily to utilize the __missing__ function,
        which serves to make the code a bit more Pythonic. The class loads
        each token individually and uses the __missing__ method to determine
        whether it is already in the dictionary (self). In the case the it
        is, the id of the particular tweet/doc is added to the relevant
        postings list.

        Query methods are detailed in their respective methods' docstrings.

        Attributes:
                all_postings: List containing ALL postings lists
                tweet_content_dict: Dictionary whose keys are twitter_ids
                        pointing to the content of the tokenized tweets.
                __current_tweet_id: ID of the doc/tweet that is currently
                        being iterated over. This is used by the __missing__ method
                        to propery organize the dictionary
        """

        STOP_WORDS = stopwords.words('english') 
        EXCLUSION_LIST = list(punctuation) + list(UNICODE_EMOJI.keys()) + ['...', 'de', 'com']

        def __init__(self, database, positional=False, in_memory=True):
                """
                Initializes by walking through each token and creating an
                inverted index as detailed above.

                :param str path: the path to the string
                """
                super().__init__()
                if type(database) is list:
                    self.database = dict(enumerate(database))
                else:
                    self.database = database
                if in_memory
                self.__indexing = False
                self.PostingsList = self._choose_postings_type(positional, in_memory)
                self._index(database)

        def __missing__(self, token: str):
                """
                If entry is missing, add posting to all_postings, then creates
                a PostingNode and sets the entry to equal that value.

                :param str token: The token to be added to the dictionary
                :return: posting
                """
                if self.__indexing:
                        self[token] = self.PostingsList([self.__curent_doc])
                        return self[token]
                else:
                        return set()

        @property
        def original_index(self):
            if issubclass(type(self.database), dict):
                return set(self.database.keys())
            else:
                return range(len(self.database))

        @staticmethod
        def _choose_postings_type(self, positional, in_memory):
            if positional:
                if in_memory:
                    return InMemoryPositionalPostings
                else:
                    return FromDiskPositionalPostings
            else:
                if in_memory:
                    return InMemoryPostingsList
                else:
                    return FromDiskPostingsList

        def __index_tokens(self, doc: list) -> None:
                """
                Walks the the list of tokens from each tweet. If a token
                is not clean, it exits without adding an entry. It then
                adds to (or uses the __missing__ method to create) the
                dictionary (self) entry for each token. It then adds to the
                posting list of each token.

                :param list tweet_content: a list of tokens from tweet
                :returns: None
                """
                for token in doc:
                    # creates entry or assigns posting_node to existing one
                    postings_list = self[token]
                    if self.__curent_doc not in postings_list:
                            # adds to end of posting list 
                            postings_list.add(self.__curent_doc)

        def get_most_freq_words(self, n: int = 10) -> List[str]:
                """
                Returns the words with the three largest frequencies.

                :param int n: the optional n number of words to return
                :return: the most frequently used words in the corpus
                :rtype: list
                """
                return heapq.nlargest(n, self, key=lambda x: self[x])

        def _index(self, database) -> None:
            # TODO: Write index to disk
            self.__indexing = True

            for doc_id, doc in database.items():
                self.__curent_doc = doc_id
                self.__index_tokens(doc)

            self.__current_doc = None
            self.__indexing = False

        def query(self, term1: str, term2: str = None) -> List[int]:
                """
                Gets the postings list for a term or the intersection or the
                posting lists of two different terms.

                :param str term1: the first (or only) string to query
                :param str term2: the optione 2nd string to intersect with
                :return: the positings list or intersection of two
                """
                if term2 is None:
                        return self[term1].postings_list
                else:
                        return list(set(self.query(term1)) & set(self.query(term2)))

        def print_query(self, term1: str, term2: str = None) -> None:
            # FIXME
            """
            Pretty prints the query method.

            :param str term1: the first term to query
            :param str term2: the optional second term to intersect with
            """
            for tweet_id in self.query(term1, term2):
                    print(f'{tweet_id}:', self.tweet_content_dict[tweet_id])

class InMemoryPostingsList(object):
    def __init__(self, postings):
        self.postings = OrderedDict({posting:0 for posting in sorted(postings)})

    def add(self, posting):
        self.postings[posting] = 0

    def as_list(self):
        return list(self.postings.keys())

    def as_array(self):
        return np.array(self.as_list())

    def update(self, postings):
        for posting in postings:
            self.add(posting)

    def __call__(self, posting):
        self.add(posting)

    def __len__(self):
        return len(self.postings)

    def __str__(self):
        return str(list(self.postings.keys()))

    def __repr__(self):
        return f'InMemoryPostingsList({str(self)})'

    def __getitem__(self, item):
        return list(self.postings.keys())[item]

def FromDiskPostingsList(InMemoryPostingsList):
