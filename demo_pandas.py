# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path

from igorwriter import utils


def demo():
    datadir = Path('./igor_pandas')
    datadir.mkdir(exist_ok=True)
    df = pd.DataFrame(
        data={
            'time': np.arange(0, 100, dtype=np.float64),
            'colInt32': np.random.randint(-100, 100, 100).astype(np.int32),
            'colUint32': np.random.randint(0, 100, 100).astype(np.uint32),
            'colSingle': np.random.randint(-100, 100, 100).astype(np.float32),
            'colDouble': np.random.randint(-100, 100, 100).astype(np.float64),
            'colComp': np.random.randint(0, 100, 100).astype(np.complex128),
        }
    )
    utils.dataframe_to_itx(df, 'igor_pandas/dataframe.itx')
    utils.dataframe_to_ibw(df, 'igor_pandas/dataframe_bin')


if __name__ == '__main__':
    demo()
