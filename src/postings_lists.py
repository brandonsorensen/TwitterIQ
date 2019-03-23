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

	def __init__(self, postings, capacity=10, dtype=np.uint32, expansion_rate=2):
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
		return f'NumericPostingsList({str(self)})'

	def __getitem__(self, item):
		return self.postings[item]
