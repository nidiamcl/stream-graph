import numpy as np
import copy


class FingerprintMeta(object):
    def __init__(self, fingerprint, rank, identifier, size, meta):
        self.fingerprint = np.array(fingerprint).astype(np.float64)
        self.rank = rank
        self.identifier = identifier
        self.size = size
        self.meta = meta

    def __key(self):
        return (self.rank, self.identifier)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, FingerprintMeta):
            return self.__key() == other.__key()
        return NotImplemented

    @classmethod
    def fromList(cls, list_meta):
        return FingerprintMeta(list_meta[0], list_meta[1], list_meta[2], list_meta[3], list_meta[4])

    def copy(self):
        return FingerprintMeta(copy.deepcopy(self.fingerprint), self.rank, self.identifier, self.size, copy.deepcopy(self.meta))

    def get_fingerprint(self):
        return self.fingerprint

    def append_to_map(self, node):
        self.meta.append(node)

    def increment(self):
        self.size = self.size + 1

    def set_fingerprint(self, fingerprint):
        self.fingerprint = np.array(fingerprint).flatten().astype(np.float64)

    def get_size(self):
        return self.size

    def get_rank(self):
        return self.rank

    def get_id(self):
        return self.identifier

    def get_meta(self):
        return self.meta

    def asList(self):
        return [self.fingerprint.tolist(), self.rank, self.identifier, self.size, self.meta] 

    def __add__(self, y):
        fingerprint = y.get_fingerprint()
        size = y.get_size()
        meta = y.get_meta()

        new_fingerprint = ((fingerprint*size)+(self.fingerprint*self.size))/(size+self.size)
        new_meta = list(set(self.meta) | set(meta))

        return FingerprintMeta(new_fingerprint, self.rank, self.identifier, size+self.size, new_meta)

    def __sub__(self, y):
        fingerprint = y.get_fingerprint()
        size = y.get_size()
        meta = y.get_meta()

        if (self.size - size) > 0:
            new_fingerprint = (self.fingerprint*self.size-fingerprint*size)/(self.size-size)
        else:
            new_fingerprint = self.fingerprint*0

        new_meta = list(set(self.meta) - set(meta))

        return FingerprintMeta(new_fingerprint, self.rank, self.identifier, self.size-size, new_meta)

    def __repr__(self):
        return '[rank: '+str(self.rank)+',id:'+ str(self.identifier)+ ',size:'+ str(self.size)+']'

    def __str__(self):
        return '[rank: '+str(self.rank)+',id:'+ str(self.identifier)+ ',size:'+ str(self.size)+']'







