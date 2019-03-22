import heapq
import numpy as np
from typing import *


class InvertedIndex(dict):
	"""
	Takes a collection of documents and creates an inverted index in which
	each element of the document, such as a token, is stored as a key in a dictionary
	with a list of identifies for the documents in which it appears serving as the
	value. Any  list of arbitrary document identifiers can be supplied in the `ids`
	argument. If no identifiers are provided, the class defaults to using a standard
	zero-indexed numeric index of the length of the provided database. Ensure that if
	custom `ids` are supplied which are non-numerical that the `numeric_ids` parameter
	is set to False. This class cannot be initialized to empty.
	"""

	def __init__(self, database: list = (), ids: list = None,
	             exclude: list = (), numeric_ids: bool = True,
	             keep_db: bool = False, keep_ids: bool = False):
		"""
		:param list-like database: a collection of elements to be indexed
		:param list-like ids: a collection of unique identifiers
		:param list-like exclude: a collection of elements to be excluded from index
		:param bool numeric_ids: whether all IDs are numeric.
		:param bool keep_db: whether to store the database as an instance variable or
							 free it from memory after indexing
		:param bool keep_ids: whether to store the IDs as an instance variable or
							  free it from memory after indexing
		"""
		super().__init__()
		if keep_db: self.database = database
		if keep_ids: self.ids = ids
		self.PostingsList = NumericPostingsList if numeric_ids else PostingsList
		self.__indexing = False
		self.__current_doc = None
		self.index(database, ids=ids, exclude=exclude)

	def __missing__(self, item: str):
		"""
		If entry is missing, add posting to all_postings, then creates
		a PostingNode and sets the entry to equal that value.

		:param str item: The token to be added to the dictionary
		:return: posting
		"""
		if self.__indexing:
			self[item] = self.PostingsList([self.__current_doc])
			return self[item]
		else:
			return []

	def _index_tokens(self, doc_body: list, doc_id: int, exclude: set) -> None:
		"""
		:param list-like doc_body: a list of tokens from tweet
		:param list-like exclude: a list of
		:returns: None
		"""
		for token in doc_body:
			# creates entry or assigns posting_node to existing one
			if token in exclude:
				continue
			self.__current_doc = doc_id
			postings_list = self[token]
			postings_list.add(doc_id)

	def n_most_common(self, n: int = 10) -> List[str]:
		"""
		Returns the words with the three largest frequencies.

		:param int n: the optional n number of words to return
		:return: the most frequently used words in the corpus
		:rtype: list
		"""
		return heapq.nlargest(n, self, key=lambda x: self[x])

	def index(self, database, ids=None, exclude=(), show_progress=0) -> None:
		if ids is None:
			if self.PostingsList is PostingsList:
				raise ValueError('Index is not numeric and no IDs are provided.')

			ids = range(len(self), len(self) + len(database))
		else:
			ids = self._check_ids(ids, database, self.PostingsList is NumericPostingsList)

		exclude = set(exclude)
		self.__indexing = True

		current_index = 0
		for doc, doc_id in zip(database, ids):
			if show_progress and not current_index % show_progress:
				print(f'Current index: {doc_id}, {current_index/len(database)}%')

			self._index_tokens(doc, doc_id, exclude)
			current_index += 1

		if self.PostingsList is NumericPostingsList:
			for postings_list in self.values():
				postings_list.finalize()

		try:
			del self.__current_doc
		except NameError:
			pass

		self.__indexing = False

	def token_count(self):
		return sum(map(len, self.values()))

	def collect_ids(self):
		try:
			return self.ids
		except AttributeError:
			pass

		all_ids = set()
		for postings_list in self.values():
			all_ids.update(postings_list)
		return sorted(all_ids)

	def query(self, term1: str, term2: str = None) -> List[int]:
		"""
		Gets the postings list for a term or the intersection or the
		posting lists of two different terms.

		:param str term1: the first (or only) string to query
		:param str term2: the optione 2nd string to intersect with
		:return: the positings list or intersection of two
		"""
		if term2 is None:
			return self[term1]
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

	def _check_ids(self, ids, database, ensure_numeric):
		"""
		Ensures that a provided list of IDs 1) is the same shape as its accompanying
		dataset, 2) contains no duplicate elements, and 3) contingent upon the boolean
		value of `ensure_numeric, converts all elements to NumPy integers.

		:param list-like ids: a list of idientifiers
		:param list-like database: a list of documents
		:param bool ensure_numeric: whether to force index to consist of numbers
		:return: list-like container of IDs
		"""
		if len(ids) != len(database):
			raise IndexError(f'The length of `ids` ({len(ids)}) does not match'
			                 f'the length of the document collection ({len(database)}).')

		# Ensures that IDs are unique
		if len(ids) != len(set(ids)) or len(ids & self.collect_ids()):
			raise IndexError('`ids` does not consist of unique elements.')

		if ensure_numeric:
			numeric_ids = np.empty((len(ids),), dtype=np.int32)
			for i in range(len(ids)):
				id_ = ids[i]
				int_id = np.int32(id_)
				if int_id != id_:
					raise IndexError(f'{id_} cannot be converted to numeric. '
					                 f'Should be {int_id}')
				numeric_ids[i] = int_id
			return numeric_ids
		else:
			return ids


class PostingsList(object):

	def __init__(self, postings):
		self.postings = set(postings)

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
		return str(sorted(list(self.postings)))

	def __repr__(self):
		return f'PostingsList({str(self)})'

	def __getitem__(self, i):
		return i in self.postings

	def __gt__(self, other):
		assert issubclass(type(other), PostingsList)
		return len(self.postings) > len(other.postings)


class NumericPostingsList(PostingsList):

	def __init__(self, postings, capacity=10, dtype=np.int32, expansion_rate=2):
		# TODO: This doesn't need to inherit from the PostingsList class.
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

	def __repr__(self):
		return f'PostingsList({str(self)})'

	def __getitem__(self, item):
		return self.postings[item]
