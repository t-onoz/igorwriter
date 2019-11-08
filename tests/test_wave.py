try:
    import unittest2 as unittest
except ImportError:
    import unittest
from tempfile import TemporaryFile

import numpy as np

from igorwriter import IgorWave, validator


class WaveTestCase(unittest.TestCase):
    def test_array_type(self):
        valid_types = (np.bool_, np.int32, np.uint32,  np.float32, np.float64, np.complex128)
        invalid_types = (np.float16, np.int64, object, str)
        array = np.random.randint(0, 100, 10)
        for vt in valid_types:
            wave = IgorWave(array.astype(vt))
            with TemporaryFile('wb') as fp:
                wave.save(fp)
            with TemporaryFile('wt') as fp:
                wave.save_itx(fp)
        for it in invalid_types:
            wave = IgorWave(array.astype(it))
            with TemporaryFile('wb') as fp:
                self.assertRaises(TypeError, wave.save, fp)
            with TemporaryFile('wt') as fp:
                self.assertRaises(TypeError, wave.save_itx, fp)

    def test_dimscale(self):
        array = np.random.randint(0, 100, 10, dtype=np.int32)
        wave = IgorWave(array)
        wave.set_dimscale('x', 0, 0.01, 's')
        with TemporaryFile('wb') as fp:
            wave.save(fp)
        with TemporaryFile('wt') as fp:
            wave.save_itx(fp)

    def test_data_units(self):
        wavename = 'wave0'
        unitslist = ('', 'sec', 'second')
        descs = ('empty units', 'short units', 'long units')
        for desc, units in zip(descs, unitslist):
            with self.subTest(desc):
                com = 'X SetScale d,.*"%s",.*\'%s\'' % (units, wavename)
                array = np.random.randint(0, 100, 10, dtype=np.int32)
                wave = IgorWave(array)
                wave.set_datascale(units)
                with TemporaryFile('wb') as fp:
                    wave.save(fp)
                with TemporaryFile('w+t') as fp:
                    wave.save_itx(fp)
                    fp.seek(0)
                    content = fp.read()
                    self.assertRegex(content, com)

    def test_dim_units(self):
        wavename = 'wave0'
        unitslist = ('', 'sec', 'second')
        descs = ('empty units', 'short units', 'long units')
        for desc, units in zip(descs, unitslist):
            with self.subTest(desc):
                com = 'X SetScale /P [xyzt],.*"%s",.*\'%s\'' % (units, wavename)
                array = np.random.randint(0, 100, 10, dtype=np.int32)
                wave = IgorWave(array)
                wave.set_dimscale('x', 0, 1, units)
                with TemporaryFile('wb') as fp:
                    wave.save(fp)
                with TemporaryFile('w+t') as fp:
                    wave.save_itx(fp)
                    fp.seek(0)
                    content = fp.read()
                    self.assertRegex(content, com)

    def test_invalid_name(self):
        name = '\'invalid_name\''  # wave cannot contain quotation marks
        array = np.random.randint(0, 100, 10, dtype=np.int32)
        wave = IgorWave(array)
        self.assertRaises(validator.InvalidNameError, wave.rename, name, on_errors='raise')
        wave.rename(name, on_errors='fix')
        self.assertEqual(wave.name, name.replace('\'', '_'))


if __name__ == '__main__':
    unittest.main()
