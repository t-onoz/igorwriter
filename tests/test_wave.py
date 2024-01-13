import unittest
import os
from pathlib import Path

import numpy as np

import igorwriter.errors
from igorwriter import IgorWave
from igorwriter.errors import TypeConversionWarning, StrippedSeriesIndexWarning

OUTDIR = Path(os.path.dirname(os.path.abspath(__file__))) / 'out'
ENCODING = "utf-8"


class WaveTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        OUTDIR.mkdir(exist_ok=True)
        for f in OUTDIR.glob('*'):
            os.remove(f)

    def test_datetime(self):
        array = np.arange(np.datetime64('2019-01-01'), np.datetime64('2019-12-31'), np.timedelta64(1, 'D'))
        wave = IgorWave(array)
        self.assertIs(wave._check_array().dtype.type, np.float64)
        with open(OUTDIR / 'datetime.ibw', 'wb') as bin:
            wave.save(bin)
        with open(OUTDIR / 'datetime.itx', 'w', encoding=ENCODING) as text:
            wave.save_itx(text)

    def test_pint(self):
        import pint
        ureg = pint.UnitRegistry()
        Q = ureg.Quantity
        a = np.arange(0.0, 10.0, 1.0)
        with self.subTest('short units'):
            wave = IgorWave(Q(a, 's'))
            bunits = wave._wave_header.dataUnits
            expected = 's'.encode(ENCODING)
            self.assertEqual(bunits, expected)
        with self.subTest('long units'):
            q = Q(a, 'kg m / s**2')
            wave = IgorWave(q)
            bunits = wave._extended_data_units
            expected = '{:~}'.format(q.units).encode(ENCODING)
            self.assertEqual(bunits, expected)

    def test_file_io(self):
        array = np.random.randint(0, 100, 10, dtype=np.int32)
        wave = IgorWave(array)
        ibw = OUTDIR / 'file_io.ibw'
        itx = OUTDIR / 'file_io.itx'

        with self.subTest('file given as Pathlib.Path object'):
            wave.save(ibw)
            wave.save_itx(itx)
        with self.subTest('file given as str'):
            wave.save(str(ibw))
            wave.save_itx(str(itx))

        with self.subTest('save in a new binary file'):
            with open(ibw, 'wb') as fp:
                wave.save(fp)
        with self.subTest('append to a binary should fail'):
            with open(ibw, 'wb') as fp:
                fp.write(b'something')
                self.assertRaises(ValueError, wave.save, fp)
            with open(ibw, 'ab') as fp:
                self.assertRaises(ValueError, wave.save, fp)
        with self.subTest('save in a truncated binary file'):
            with open(ibw, 'wb') as fp:
                wave.save(fp)

    def test_array_type(self):
        valid_types = (np.bool_, np.float16, np.int32, np.uint32, np.int64, np.uint64, np.float32, np.float64, np.complex128, np.object_, np.str_, np.bytes_, int, float)
        for vt in valid_types:
            with self.subTest('type: %r' % vt):
                wave = IgorWave(np.random.randint(0, 100, 10).astype(vt))
                with open(OUTDIR / 'array_type_{}.ibw'.format(vt.__name__), 'wb') as fp:
                    wave.save(fp)
                with open(OUTDIR / 'array_type_{}.itx'.format(vt.__name__), 'w') as fp:
                    wave.save_itx(fp)

    def test_bool_to_itx(self):
        a = np.array([True, True, True, True, True], dtype=np.bool_)
        w = IgorWave(a, 'boolwave')
        with open(OUTDIR / 'bool_to_itx.itx', 'w+t') as fp:
            w.save_itx(fp)
            fp.seek(0)
            self.assertRegex(fp.read(), r'BEGIN\n1')

    def test_int64(self):
        # int64 and uint64 overflow
        a = np.array([2**63 - 1]*10, dtype=np.int64)
        an = -a
        au = np.array([2**63 - 1]*10, dtype=np.uint64)
        self.assertRaises(OverflowError, IgorWave(a).save, OUTDIR/'int64_overflow.ibw')
        self.assertRaises(OverflowError, IgorWave(an).save, OUTDIR/'int64_overflow.ibw')
        self.assertRaises(OverflowError, IgorWave(au).save, OUTDIR/'int64_unsigned_overflow.ibw')
        w = IgorWave(a, int64_support=True)
        w.save(OUTDIR/'int64.ibw')
        self.assertEqual(w._wave_header.type, 0x80)
        com = 'WAVES /L'
        with open(OUTDIR / 'int64.itx', 'w+t', encoding='utf-8') as fp:
            w.save_itx(fp)
            fp.seek(0)
            content = fp.read()
            self.assertRegex(content, com)
        w = IgorWave(au, int64_support=True)
        w.save(OUTDIR/'int64_unsigned.ibw')
        self.assertEqual(w._wave_header.type, 0x80+0x40)
        com = 'WAVES /U/L'
        with open(OUTDIR / 'int64_unsigned.itx', 'w+t', encoding='utf-8') as fp:
            w.save_itx(fp)
            fp.seek(0)
            content = fp.read()
            self.assertRegex(content, com)

    def test_dimscale(self):
        array = np.random.randint(0, 100, 10, dtype=np.int32)
        wave = IgorWave(array)
        wave.set_dimscale('x', 0, 0.01, 's')
        with open(OUTDIR / 'dimscale.ibw', 'wb') as fp:
            wave.save(fp)
        with open(OUTDIR / 'dimscale.itx', 'w') as fp:
            wave.save_itx(fp)

    def test_data_units(self):
        wavename = 'wave0'
        unitslist = ('', 'sec', 'second')
        descs = ('empty', 'short', 'long')
        for desc, units in zip(descs, unitslist):
            with self.subTest(desc):
                com = 'X SetScale d,.*"%s",.*\'%s\'' % (units, wavename)
                array = np.random.randint(0, 100, 10, dtype=np.int32)
                wave = IgorWave(array)
                wave.set_datascale(units)
                with open(OUTDIR / 'data_units_{}.ibw'.format(desc), 'wb') as fp:
                    wave.save(fp)
                with open(OUTDIR / 'data_units_{}.itx'.format(desc), 'w+t') as fp:
                    wave.save_itx(fp)
                    fp.seek(0)
                    content = fp.read()
                    self.assertRegex(content, com)

    def test_dim_units(self):
        wavename = 'wave0'
        unitslist = ('', 'sec', 'second')
        descs = ('empty', 'short', 'long')
        for desc, units in zip(descs, unitslist):
            with self.subTest(desc):
                com = 'X SetScale /P [xyzt],.*"%s",.*\'%s\'' % (units, wavename)
                array = np.random.randint(0, 100, 10, dtype=np.int32)
                wave = IgorWave(array)
                wave.set_dimscale('x', 0, 1, units)
                with open(OUTDIR / 'dim_units_{}.ibw'.format(desc), 'wb') as fp:
                    wave.save(fp)
                with open(OUTDIR / 'dim_units_{}.itx'.format(desc), 'w+t') as fp:
                    wave.save_itx(fp)
                    fp.seek(0)
                    content = fp.read()
                    self.assertRegex(content, com)

    def test_note(self):
        wavename = 'wave_with_note'
        note = 'A\nB\nC'
        com = 'X Note \'wave_with_note\', "A\\nB\\nC"'
        wave = IgorWave([1, 2, 3], name=wavename)
        wave.set_note(note)
        with open(OUTDIR / 'note.ibw', 'wb') as fp:
            wave.save(fp)
        with open(OUTDIR / 'note.itx', 'w+t') as fp:
            wave.save_itx(fp)
            fp.seek(0)
            content = fp.read()
            self.assertIn(com, content)

    def test_invalid_name(self):
        name = '\'invalid_name\''  # wave cannot contain quotation marks
        array = np.random.randint(0, 100, 10, dtype=np.int32)
        wave = IgorWave(array)
        self.assertRaises(igorwriter.errors.InvalidNameError, wave.rename, name, on_errors='raise')
        self.assertWarns(igorwriter.errors.RenameWarning, wave.rename, name, on_errors='fix')
        self.assertEqual(wave.name, name.replace('\'', '_'))

    def test_multiwave_itx(self):
        a = np.random.random(size=2*3*4*2)
        w1 = IgorWave(a, 'w_1d')
        w2 = IgorWave(a.reshape((2, -1)), 'w_2d')
        w3 = IgorWave(a.reshape((2, 3, -1)), 'w_3d')
        w4 = IgorWave(a.reshape((2, 3, 4, -1)), 'w_4d')
        with open(OUTDIR / 'multiwave_itx.itx', 'w+t') as fp:
            w1.save_itx(fp)
            w2.save_itx(fp)
            w3.save_itx(fp)
            w4.save_itx(fp)

    def test_dimlabel(self):
        a = np.random.random(size=(2, 3, 4, 5))
        wavename = 'dimlabeltest'
        with self.subTest('invalid dimlabel'):
            w = IgorWave(a, wavename)
            self.assertRaises(igorwriter.errors.InvalidNameError, w.set_dimlabel, 0, 0, "'")
            w.set_dimlabel(0, 0, "a"*31)
            self.assertRaises(ValueError, w.set_dimlabel, 0, 0, "a"*32)
        with self.subTest('dimlabel for entire row'):
            label = 'rowname'
            w = IgorWave(a, wavename)
            w.set_dimlabel(0, -1, label)
            com = "X SetDimLabel 0,-1,'%s','%s'" % (label, wavename)
            with open(OUTDIR / 'dimlabel_row_entire.itx', 'w+t') as fp:
                w.save_itx(fp)
                fp.seek(0)
                self.assertRegex(fp.read(), com)
            with open(OUTDIR / 'dimlabel_row_entire.ibw', 'wb') as fp:
                w.save(fp)
        with self.subTest('dimlabel for row element'):
            label = 'row0'
            w = IgorWave(a, wavename)
            w.set_dimlabel(0, 0, label)
            com = "X SetDimLabel 0,0,'%s','%s'" % (label, wavename)
            with open(OUTDIR / 'dimlabel_row_elm.itx', 'w+t') as fp:
                w.save_itx(fp)
                fp.seek(0)
                self.assertRegex(fp.read(), com)
            with open(OUTDIR / 'dimlabel_row_elm.ibw', 'wb') as fp:
                w.save(fp)
        with self.subTest('multiple dimlabels'):
            labels = {-1: 'rowname', 0: 'row0', 1: 'row1'}
            w = IgorWave(a, wavename)
            for i, l in labels.items():
                w.set_dimlabel(0, i, l)
            coms = ["X SetDimLabel 0,%d,'%s','%s'" % (i, l, wavename) for i, l in labels.items()]
            with open(OUTDIR / 'dimlabel_multi.itx', 'w+t') as fp:
                w.save_itx(fp)
                fp.seek(0)
                string = fp.read()
                for com in coms:
                    self.assertRegex(string, com)
            with open(OUTDIR / 'dimlabel_multi.ibw', 'wb') as fp:
                w.save(fp)
        with self.subTest('dimlabel for columns'):
            label = 'colname'
            w = IgorWave(a, wavename)
            w.set_dimlabel(1, -1, label)
            com = "X SetDimLabel 1,-1,'%s','%s'" % (label, wavename)
            with open(OUTDIR / 'dimlabel_col_entire.itx', 'w+t') as fp:
                w.save_itx(fp)
                fp.seek(0)
                self.assertRegex(fp.read(), com)
            with open(OUTDIR / 'dimlabel_col_entire.ibw', 'wb') as fp:
                w.save(fp)
        with self.subTest('dimlabel deletion'):
            w = IgorWave(a, wavename)
            w.set_dimlabel(0, -1, 'test')
            self.assertEqual(w._bin_header.dimLabelsSize[0], 32)
            self.assertTrue(w._dimension_labels[0])
            w.set_dimlabel(0, -1, '')
            self.assertEqual(w._bin_header.dimLabelsSize[0], 0)
            self.assertFalse(w._dimension_labels[0])


    def test_textwave(self):
        a = np.array(['a', 'bb', 'ccc', 'dddd', 'eeeee', 'ffffff'])
        w = IgorWave(a, 'mytextwave')
        with open(OUTDIR / 'textwave.itx', 'w+t') as fp:
            w.save_itx(fp)
            fp.seek(0)
            string = fp.read()
            self.assertRegex(string, r'"a"')
        with open(OUTDIR / 'textwave.ibw', 'wb') as fp:
            w.save(fp)

    def test_textwave_from_bytes(self):
        a = np.array([b'a', b'bb', b'ccc', b'dddd', b'eeeee', b'ffffff'])
        w = IgorWave(a, 'mytextwave')
        with open(OUTDIR / 'textwave_from_bytes.itx', 'w+t') as fp:
            w.save_itx(fp)
            fp.seek(0)
            string = fp.read()
            self.assertRegex(string, r'"a"')
        with open(OUTDIR / 'textwave_from_bytes.ibw', 'wb') as fp:
            w.save(fp)

    def test_textwave_multidim(self):
        a = np.array([str(x)*np.random.randint(1, 10) for x in 'abcdefghijklmnop']).reshape((2, 2, 2, 2))
        w = IgorWave(a, 'my4dtextwave')
        with open(OUTDIR / 'textwave_multidim.itx', 'w+t') as fp:
            w.save_itx(fp)
        with open(OUTDIR / 'textwave_multidim.ibw', 'wb') as fp:
            w.save(fp)

    def test_textwave_special_chars(self):
        a = np.array(['a\tb', 'a\rb', 'a\nb', 'a\r\nb', 'a\'b', 'a"b', 'a\\b'])
        w = IgorWave(a, 'mytextwave')
        with open(OUTDIR / 'textwave_spchars.itx', 'w+t') as fp:
            w.save_itx(fp)
            fp.seek(0)
            string = fp.read()
            self.assertTrue('\\t' in string)
            self.assertTrue('\\r' in string)
            self.assertTrue('\\n' in string)
            self.assertTrue('\\\'' in string)
            self.assertTrue('\\"' in string)
            self.assertTrue('\\\\' in string)
        with open(OUTDIR / 'textwave_spchars.ibw', 'wb') as fp:
            w.save(fp)

    def test_optional_data(self):
        # wave with a lot of optional data
        a = np.array([str(x)*np.random.randint(1, 10) for x in 'abcdefghijklmnop']).reshape((2, 2, 2, 2))
        w = IgorWave(a, 'optDataTest')
        w.set_dimscale('x', 0, 1, 'verylongdimunitX')
        w.set_dimscale('y', 0, 1, 'verylongdimunitY')
        w.set_dimscale('z', 0, 1, 'verylongdimunitZ')
        w.set_dimscale('t', 0, 1, 'verylongdimunitT')
        w.set_datascale('verylongunitData')
        w.set_dimlabel(0, -1, 'dimensionlabel0')
        w.set_dimlabel(1, -1, 'dimensionlabel1')
        w.set_dimlabel(2, -1, 'dimensionlabel2')
        w.set_dimlabel(3, -1, 'dimensionlabel3')
        with open(OUTDIR / 'optional_data.ibw', 'wb') as fp:
            w.save(fp)
        with open(OUTDIR / 'optional_data.itx', 'w') as fp:
            w.save_itx(fp)

    def test_object_array(self):
        # wave with a lot of optional data
        with self.subTest('mixed type array'):
            a = np.array(['a', 1.0, 0.5, b'spam'], dtype=object)
            w = IgorWave(a, 'mixed_dtype')
            with open(OUTDIR / 'object_array_mixed_dtype.itx', 'w+t') as fp:
                self.assertWarns(TypeConversionWarning, w.save_itx, fp)
                fp.seek(0)
                text = fp.read()
                self.assertTrue('WAVES /T' in text)
        with self.subTest('numeric object array'):
            a = np.array([0.1, 0.2, 0.3], dtype=object)
            w = IgorWave(a, 'numeric_object')
            with open(OUTDIR / 'object_array_numeric.itx', 'w+t') as fp:
                self.assertWarns(TypeConversionWarning, w.save_itx, fp)
                fp.seek(0)
                text = fp.read()
                self.assertTrue('WAVES /D' in text)

    def test_unicode_wave(self):
        a = np.array(['Hello', '你好', 'こんにちは', '안녕하세요'], dtype=np.str_)
        w = IgorWave(a, 'H你こ안', unicode=True)
        w.save(OUTDIR / 'unicode_wave.ibw')
        self.assertEqual(w._wave_header.waveNameEncoding, 1)
        self.assertEqual(w._wave_header.waveUnitsEncoding, 1)
        self.assertEqual(w._wave_header.waveNoteEncoding, 1)
        self.assertEqual(w._wave_header.waveDimLabelEncoding, 1)
        self.assertEqual(w._wave_header.textWaveContentEncoding, 1)
        w.save_itx(OUTDIR / 'unicode_wave.itx')
        with open(OUTDIR / 'unicode_wave.itx', encoding='utf-8') as f:
            f.read()

    def test_pandas_series(self):
        import pandas as pd
        with self.subTest('simple Series object'):
            s = pd.Series([1, 2, 3, 4, 5, 6])
            w = IgorWave(s, 'serieswave')
        with self.subTest('Series object with uniform Index'):
            s = pd.Series([1, 2, 3, 4, 5, 6])
            s.index = np.linspace(1, 2, 6)
            w = IgorWave(s, 'series_uniformind')
            self.assertAlmostEqual(w._wave_header.sfA[0], 0.2)
            self.assertAlmostEqual(w._wave_header.sfB[0], 1.0)
        with self.subTest('Series object with invalid Index names'):
            s = pd.Series([1, 2, 3, 4, 5, 6])
            s.index = np.linspace(1, 2, 6)
            for n in ('\'', 'a'*32):
                s.index.name = n
                with self.assertWarns(UserWarning):
                    w = IgorWave(s)
                self.assertAlmostEqual(w._wave_header.sfA[0], 0.2)
                self.assertAlmostEqual(w._wave_header.sfB[0], 1.0)
                self.assertNotIn(-1, w._dimension_labels[0])
        with self.subTest('Series object with uniform Index with a name'):
            s.index.name = 'myindex'
            w = IgorWave(s, 'series_uniformind')
            self.assertEqual(w._dimension_labels[0][-1], b'myindex')
        with self.subTest('Series object with nonuniform Index'):
            s = pd.Series([1, 2, 3])
            s.index = [1, 2, 3.05]
            s.index.name = 'myindex'
            with self.assertWarns(StrippedSeriesIndexWarning):
                w = IgorWave(s, 'series_nonuniformind')
            # when index is invalid, dimension labels should be empty
            self.assertAlmostEqual(w._wave_header.sfA[0], 1)
            self.assertAlmostEqual(w._wave_header.sfB[0], 0)
            self.assertNotIn(-1, w._dimension_labels[0])
        with self.subTest('Series object with incompatible dtype Index'):
            s = pd.Series([1, 2, 3])
            s.index = [1, 2, 'spam']
            s.index.name = 'myindex'
            with self.assertWarns(StrippedSeriesIndexWarning):
                w = IgorWave(s, 'series_invalidind')
            # when index is invalid, dimension labels should be empty
            self.assertAlmostEqual(w._wave_header.sfA[0], 1)
            self.assertAlmostEqual(w._wave_header.sfB[0], 0)
            self.assertNotIn(-1, w._dimension_labels[0])
        with self.subTest('pint-pandas extended types'):
            import pint_pandas
            u = "kg m / s**2"
            s = pd.Series([1.0, 2.0, 2.0, 3.0], dtype=f"pint[{u}]")
            expected = '{:~}'.format(s.pint.units).encode(ENCODING)
            wave = IgorWave(s)
            bunits = wave._extended_data_units
            self.assertEqual(bunits, expected)


if __name__ == '__main__':
    unittest.main()
