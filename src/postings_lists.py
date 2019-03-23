import numpy as np


class CustomPostingsList(object):

	# TODO: Enforce uniform type
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
		assert issubclass(type(other), CustomPostingsList)
		return len(self.postings) > len(other.postings)


class NumericPostingsList(CustomPostingsList):
	"""
	A postings list that consists only of numbers. The postings are stored
	in 1-dimensional NumPy arrays not as their real values but as the difference
	between the value and the value directly before it. When the contents of the
	postings list are printed, they are then converted back to their real values.
	"""

	def __init__(self, postings: list, capacity: int = 10, dtype: type = np.uint32,
	             expansion_rate: float = 2.0):
		"""
		Creates a postings list consisting only of numeric values and then compresses
		them to the difference between each consecutive value.

		:param list postings: a list of postings
		:param int capacity: the initial capacity of the underlying array, default 10
		:param type dtype: the data type of the array
		:param float expansion_rate: the rate at which the underlying array grows
			   when its capacity is met.
		"""
		super().__init__(postings)
		self.highest = None
		assert issubclass(dtype, np.unsignedinteger)
		self.dtype = dtype
		self.expansion_rate = expansion_rate

		if len(postings) > capacity:
			self.highest = postings[-1]
			self.postings = np.array(list(self.compress(postings)), dtype=dtype)
			self.capacity = self.size = len(postings)
		else:
			self.size = 0
			self.capacity = capacity
			self.postings = np.empty((capacity,), dtype=dtype)
			self.update(postings)

	def add(self, posting):
		posting_ = self.dtype(posting)

		if self.size == self.capacity:
			self._extend_array()

		if self.highest == posting_:
			return
		self.postings[self.size] = posting_
		self.highest = posting_
		self.size += 1

	def _extend_array(self):
		"""Extends the capacity of the array."""
		self.capacity = int(self.capacity * self.expansion_rate)
		new_array = np.empty((self.capacity,), dtype=self.dtype)
		new_array[:self.size] = self.postings
		self.postings = new_array

	def finalize(self):
		"""
		Reduces the length of the underlying array to the current size of
		the postings list, removing reference to the larger array.
		"""
		self.postings = self.postings[:self.size]
		self.capacity = self.size

	@staticmethod
	def compress(num_array):
		"""Converts an array """
		for i in range(len(num_array)):
			if not i:
				yield num_array[i]
			else:
				yield num_array[i] - num_array[i - 1]

	@staticmethod
	def decompress(num_array):
		for i in range(len(num_array)):
			if not i:
				yield num_array[i]
			else:
				yield num_array[i] + num_array[i - 1]

	def update(self, postings):
		for posting in postings:
			self.add(posting)

	def __str__(self):
		return str(list(self.decompress(self.postings[:self.size])))

	def __repr__(self):
		return f'NumericPostingsList({str(self)})'

	def __getitem__(self, item):
		return self.postings[item]
