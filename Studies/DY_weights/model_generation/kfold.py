import numpy as np
import pandas as pd


class KFolder:
    def __init__(self, k):
        self.k = k
        self.selection_column = "FullEventId"

    def split(self, df, k):
        idx = df[self.selection_column].values.copy()
        idx = np.mod(idx, k)
        for i in range(k):
            mask = idx == i
            yield df[mask].copy(), df[~mask].copy()
