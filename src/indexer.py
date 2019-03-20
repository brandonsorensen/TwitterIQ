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
	"""

	def __init__(self, database, ids=None, exclude=(), numeric_ids=True):
		"""
		Initializes by walking through each token and creating an
		inverted index as detailed above.

		:param str path: the path to the string
		"""
		super().__init__()
		self.database = database

		if ids is None:
			self.ids = np.array(range(len(database)), dtype=np.int32)
		else:
			if len(ids) != len(database):
				raise IndexError(f'The length of `ids` ({len(ids)}) does not match'
				                 f'the length of the document collection ({len(database)}).')

			self.database = database
			self.ids = ids

		self.PostingsList = NumericPostingsList if numeric_ids else PostingsList
		self.__indexing = False
		self._index(database, set(exclude))
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

	def __index_tokens(self, doc: list, exclude: list) -> None:
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
			if token in exclude:
				continue
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

	def _index(self, database, exclude, ids=None) -> None:
		if ids is None:
			if self.PostingsList is PostingsList:
				raise ValueError('Index is not numeric and no IDs are provided.')

			ids = range(len(self), len(self) + len(database))

		self.__indexing = True

		for doc_id, doc in zip(ids, database):
			self.__curent_doc = doc_id
			self.__index_tokens(doc, exclude)

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
		if type(other) in [list, np.array]:
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

	def __gt__(self, other):
		assert issubclass(type(other), PostingsList)
		return len(self.postings) > len(other.postings)


class NumericPostingsList(PostingsList):

	def __init__(self, postings, capacity=10, dtype=np.int32, expansion_rate=2):
		super().__init__(postings)
		self.size = len(self.postings)
		self.first = postings[0] if self.size else None
		self.dtype = dtype
		assert issubclass(self.dtype, np.number)
		self.expansion_rate = expansion_rate

		if self.size > capacity:
			self.postings = np.array(list(self.compress(self.postings)),
			                         dtype=dtype)
			self.capacity = self.size
		else:
			self.capacity = capacity
			self.postings = np.empty((capacity,), dtype=dtype)
			self.update(postings)

	def add(self, posting):
		posting_ = self.dtype(posting)
		if self.size == 0:
			self.first = posting_

		if self.size == self.capacity:
			self._extend_array()

		compressed = posting_ - self.first
		if self.postings[self.size - 1] == compressed:
			return
		self.postings[self.size] = compressed
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
				yield int(num_array[i] + self.first)

	def __str__(self):
		return str(list(self.decompress(self.postings[:self.size])))
