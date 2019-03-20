import heapq
import numpy as np
from typing import *


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

	def __init__(self, database, ids=None, exclude=None, numeric_ids=True):
		"""
		Initializes by walking through each token and creating an
		inverted index as detailed above.

		:param str path: the path to the string
		"""
		super().__init__()
		self.database = database

		if ids is None:
			self.ids = np.ndarray(range(len(database)), dtype=np.int32)
		else:
			if len(ids) != len(database):
				raise IndexError(f'The length of `ids` ({len(ids)}) does not match'
				                 f'the length of the document collection ({len(database)}).')

			self.database = database
			self.ids = ids

		self.PostingsList = NumericPostingsList if numeric_ids else PostingsList
		self.__indexing = False
		self._index(database)
		for token in exclude:
			del self[token]
		self.vocab = self.keys()

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

	def _index(self, database, ids=None) -> None:
		if ids is None:
			if self.PostingsList is PostingsList:
				raise ValueError('Index is not numeric and no IDs are provided.')

			ids = range(len(self), len(self) + len(database))

		self.__indexing = True

		for doc_id, doc in zip(ids, database):
			self.__curent_doc = doc_id
			self.__index_tokens(doc)

		if self.PostingsList is NumericPostingsList:
			for postings_list in self.values():
				postings_list.finalize()

		self.__current_doc = None
		self.__indexing = False

	def token_count(self):
		return sum(map(len, self.values()))

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


class PostingsList(object):

	def __init__(self, postings):
		self.postings = postings

	def add(self, posting):
		self.postings.add(posting)

	def as_array(self):
		return np.array(self.postings)

	def update(self, postings):
		for posting in postings:
			self.add(posting)

	def __call__(self, posting):
		self.add(posting)

	def __add__(self, other):
		if type(other) in [list, np.ndarray]:
			self.update(other)

		else:
			self.add(other)

	def __len__(self):
		return len(self.postings)

	def __str__(self):
		return str(self.postings)

	def __repr__(self):
		return f'PostingsList({str(self)})'

	def __getitem__(self, i):
		return self.postings[i]


class NumericPostingsList(PostingsList):

	def __init__(self, postings, capacity=30, dtype=np.int32, expansion_rate=2):
		super().__init__(postings)
		self.size = len(self.postings)
		self.first = postings[0] if self.size else None

		if self.size > capacity:
			self.postings = np.array(list(self.compress(self.postings)),
			                         dtype=dtype)
			self.capacity = self.size
		else:
			self.capacity = capacity
			self.postings = np.empty((capacity,))
			self.update(self.postings)

		self.expansion_rate = expansion_rate

	def add(self, posting):
		if type(posting) is not int or issubclass(type(posting), np.integer):
			raise TypeError('Posting must be of type int.')

		if self.size == 0:
			self.first = posting

		if self.size == self.capacity:
			self._extend_array()

		self.postings[self.size] = posting - self.first
		self.size += 1

	def _extend_array(self):
		self.capacity *= self.expansion_rate
		new_array = np.empty((self.capacity,))
		new_array[:self.size] = self.postings
		self.postings = new_array

	def finalize(self):
		self.postings = self.postings[:self.size]
		self.capacity = self.size

	def compress(self, num_array):
		for i in range(len(num_array)):
			if not i:
				yield self.first
			else:
				yield num_array[i] - self.first

	def decompress(self, num_array):
		for i in range(len(num_array)):
			if not i:
				yield self.first
			else:
				yield num_array[i] + self.first

	def __str__(self):
		return str(list(self.decompress(self.postings)))
