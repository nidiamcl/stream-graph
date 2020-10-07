import numpy as np
import copy

def broadcast_merge(bmeta, fingerprints_meta):
    fp_rank = fp[0][1]
    fp_id =fp[0][2]
    fp_size = fp[0][3]
    fp_map = fp[0][4]
    fp = np.array(fp[0][0])

    sizes = get_field(fingerprints_meta, 'size')
    fps = get_field(fingerprints_meta, 'fp')
    fmaps = get_field(fingerprints_meta, 'fmap')

    size_diff = [s-fp_size for s in sizes]

    fps_weighted = [(sizes[i]*(np.array(fps[i])))-(fp*fp_size) for i in range(len(fps))] 
    fps_sum = (fp_size*fp).copy()
    for fp_tmp in fps_weighted:
        fps_sum += fp_tmp

    size = sum([size_diff[i] for i in range(len(fps))]) + fp_size

    new_fmaps = []
    for fm in fmaps:
        new_fmaps += fm
    new_fmaps = list(set(new_fmaps + fp_map))

    return [fps_sum/size, fp_rank, fp_id, size, new_fmaps]

class FingerprintMeta(object):
    def __init__(self, fingerprint, rank, identifier, size, meta):
        self.fingerprint = np.array(fingerprint)
        self.rank = rank
        self.identifier = identifier
        self.size = size
        self.meta = meta

    @classmethod
    def fromList(cls, list_meta):
        return cls(list_meta[0], list_meta[1], list_meta[2], list_meta[3], list_meta[4]

    def copy(self):
        return cls(copy.deepcopy(self.fingerprint), self.rank, self.identifier, self.size, copy.deepcopy(self.meta)

    def get_fingerprint(self):
        return self.fingerprint

    def append_to_map(self, node):
        self.meta.append(node)

    def increment(self):
        self.size = self.size + 1

    def set_fingerprint(self, fingerprint):
        self.fingerprint = np.array(fingerprint).flatten()

    def get_size(self):
        return self.size

    def get_rank(self):
        return self.rank

    def get_id(self):
        return self.identifier

    def get_meta(self):
        return self.meta

    def asList(self):
        return [self.fingerprint, self.rank, self.identifier, self.size, self.meta] 

    def __add__(self, y):
        fingerprint = y.get_fingerprint()
        size = y.get_size()
        meta = y.get_meta()

        new_fingerprint = (fingerprint*size+self.fingerprint*self.size)/(size+self.size)
        new_meta = list(set(self.meta + meta))

        return FingerprintMeta(new_fingerprint, self.rank, self.identifier, size+self.size, new_meta)

    def __sub__(self, y):
        fingerprint = y.get_fingerprint()
        size = y.get_size()
        meta = y.get_meta()

        new_fingerprint = (self.fingerprint*self.size-fingerprint*size)/(self.size-size)
        new_meta = list(set(self.meta + meta))

        return FingerprintMeta(new_fingerprint, self.rank, self.identifier, self.size-size, new_meta)









