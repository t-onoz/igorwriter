import unittest
from tempfile import TemporaryFile, TemporaryDirectory

import numpy as np

from igorwriter import utils


class UtilsTestCase(unittest.TestCase):
    def test_pandas(self):
        import pandas as pd
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
        with TemporaryDirectory() as datadir:
            utils.dataframe_to_itx(df, datadir + 'df.itx')
            with open(datadir + 'df_fp.itx', 'w') as fp:
                utils.dataframe_to_itx(df, fp)
            utils.dataframe_to_ibw(df, datadir + 'df_bin')


if __name__ == '__main__':
    unittest.main()
