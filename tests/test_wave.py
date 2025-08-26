import unittest
import os
import warnings
from pathlib import Path

import numpy as np

import igorwriter.errors
from igorwriter import IgorWave
from igorwriter.errors import TypeConversionWarning, StrippedSeriesIndexWarning
from igorwriter.igorwave import  TYPES, ITX_TYPES

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
        converted = wave._check_array()
        expected = (array - np.datetime64('1904-01-01 00:00:00')) / np.timedelta64(1, 's')
        self.assertIs(converted.dtype.type, np.float64)
        np.testing.assert_allclose(converted, expected)
        self.assertEqual(wave._wave_header.dataUnits, b'dat')
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

    def test_builtin_numeric_types(self):
        array = np.arange(10)
        for type_ in set(np.sctypeDict.values()):
            if np.issubdtype(type_, np.number):
                with self.subTest(f'type: {type_.__name__}'):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=TypeConversionWarning)
                        w = IgorWave(array.astype(type_))
                        outarray = w._cast_array()
                        self.assertIn(outarray.dtype.type, TYPES)
                        np.testing.assert_allclose(array, outarray)

    def test_types(self):
        array = np.arange(10)
        for type_ in TYPES:
            with self.subTest(f'type: {type_.__name__}'):
                w_ibw = IgorWave(array.astype(type_), int64_support=True)
                w_ibw.save(OUTDIR / f'types_{type_.__name__}.ibw')
                self.assertEqual(w_ibw._wave_header.type, TYPES[type_])
                w_itx = IgorWave(array.astype(type_), int64_support=True)
                with open(OUTDIR / f'types_{type_.__name__}.itx', mode='w+t') as fp:
                    w_itx.save_itx(fp)
                    fp.seek(0)
                    contents = fp.read()
                self.assertIn(f'WAVES {ITX_TYPES[type_]}', contents)

    def test_complex(self):
        a = np.array([[1+2j, 3+4j, 5+6j], [7+8j, 9+10j, 11+12j]], dtype=np.complex128)
        a_bytes = np.array([1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12], dtype=np.float64).tobytes()
        w = IgorWave(a, 'complexwave')
        with open(OUTDIR / 'complex_wave.itx', 'w+t') as fp:
            w.save_itx(fp)
            fp.seek(0)
            contents = fp.read()
        self.assertIn('WAVES /C/D', contents)
        self.assertIn('1.0\t2.0', contents)
        with open(OUTDIR / 'complex_wave.ibw', 'w+b') as fp:
            w.save(fp)
            fp.seek(0)
            contents = fp.read()
        self.assertIn(a_bytes, contents)

    def test_bool_to_itx(self):
        a = np.array([True, True, True, True, True], dtype=np.bool_)
        w = IgorWave(a, 'boolwave')
        with open(OUTDIR / 'bool_to_itx.itx', 'w+t') as fp:
            self.assertWarns(TypeConversionWarning, w.save_itx, fp)
            fp.seek(0)
            self.assertRegex(fp.read(), r'BEGIN\n1')

    def test_int64(self):
        # int64 and uint64 overflow
        a = np.array([2**63 - 1]*10, dtype=np.int64)
        an = -a
        au = np.array([2**63 - 1]*10, dtype=np.uint64)
        wa = IgorWave(a)
        wan = IgorWave(an)
        wau = IgorWave(au)
        self.assertWarns(TypeConversionWarning, wa.save, OUTDIR/'int64_overflow.ibw')
        self.assertWarns(TypeConversionWarning, wan.save, OUTDIR/'int64_overflow2.ibw')
        self.assertWarns(TypeConversionWarning, wau.save, OUTDIR/'int64_unsigned_overflow.ibw')
        self.assertEqual(wa._array_saved.dtype, 'float64')
        self.assertEqual(wan._array_saved.dtype, 'float64')
        self.assertEqual(wau._array_saved.dtype, 'float64')

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

    def test_multidim(self):
        array = np.zeros((2, 3, 4, 5), dtype=np.object_)
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    for l in range(5):
                        array[i, j, k, l] = f'({i},{j},{k},{l})'
        array = array.astype(np.str_)
        w_1d = IgorWave(array[:, 0, 0, 0], 'multidim_1d')
        w_1d.save(OUTDIR / 'multidim_1d.ibw')
        w_1d.save_itx(OUTDIR / 'multidim_1d.itx')
        w_2d = IgorWave(array[:, :, 0, 0], 'multidim_2d')
        w_2d.save(OUTDIR / 'multidim_2d.ibw')
        w_2d.save_itx(OUTDIR / 'multidim_2d.itx')
        w_3d = IgorWave(array[:, :, :, 0], 'multidim_3d')
        w_3d.save(OUTDIR / 'multidim_3d.ibw')
        w_3d.save_itx(OUTDIR / 'multidim_3d.itx')
        w_4d = IgorWave(array[:, :, :, :], 'multidim_4d')
        w_4d.save(OUTDIR / 'multidim_4d.ibw')
        w_4d.save_itx(OUTDIR / 'multidim_4d.itx')

    def test_dimscale(self):
        array = np.random.randint(0, 100, 10, dtype=np.int32)
        wave = IgorWave(array)
        wave.set_dimscale('x', 1, 0.01, 's')
        self.assertAlmostEqual(wave._wave_header.sfA[0], 0.01)
        self.assertAlmostEqual(wave._wave_header.sfB[0], 1.0)
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
        wave = IgorWave([1.0, 2.0, 3.0], name=wavename)
        wave.set_note(note)
        with open(OUTDIR / 'note.ibw', 'wb') as fp:
            wave.save(fp)
        with open(OUTDIR / 'note.itx', 'w+t') as fp:
            wave.save_itx(fp)
            fp.seek(0)
            content = fp.read()
            self.assertIn(com, content)

    def test_formula(self):
        wavename = 'sinwave'
        formula = 'sin(x)'
        wave = IgorWave(np.zeros(101), name=wavename)
        wave.set_formula(formula)
        self.assertEqual(wave._bin_header.formulaSize, len(formula)+1)
        wave.set_note('sine wave')
        wave.set_dimscale('x', 0, np.pi*2/100)
        com = 'X SetFormula \'sinwave\', "sin(x)"'
        with open(OUTDIR / 'formula.ibw', 'w+b') as fp:
            wave.save(fp)
            fp.seek(0)
            content = fp.read()
            self.assertIn(b'sin(x)\x00sine wave', content)
        with open(OUTDIR / 'formula.itx', 'w+t') as fp:
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
        a = np.arange(10, dtype=np.int32)
        w1 = IgorWave(a, 'w1')
        w2 = IgorWave(a, 'w2')
        w3 = IgorWave(a, 'w3')
        w4 = IgorWave(a, 'w4')
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
            self.assertRaises(igorwriter.errors.InvalidNameError, w.set_dimlabel, 0, 0, "a"*32)
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
        with open(OUTDIR / 'textwave_spchars.ibw', 'w+b') as fp:
            w.save(fp)
            fp.seek(0)
            content = fp.read()
            self.assertIn(b''.join(e.encode(w._encoding) for e in a), content)

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

    def test_image_mode(self):
        a = np.arange(1*2*3*4, dtype=np.int32)
        # 1d
        w = IgorWave(a, 'imagewave1d')
        np.testing.assert_allclose(a, w._check_array(image=False))
        np.testing.assert_allclose(a, w._check_array(image=True))
        w.save_itx(OUTDIR / 'imgwave_1d_norm.itx')
        w.save_itx(OUTDIR / 'imgwave_1d_img.itx', image=True)
        w.save(OUTDIR / 'imgwave_1d_norm.ibw')
        w.save(OUTDIR / 'imgwave_1d_img.ibw', image=True)
        # 2d
        a_2d = a.reshape((6, 4))
        w = IgorWave(a_2d, 'imagewave2d')
        np.testing.assert_allclose(a_2d, w._check_array(image=False))
        np.testing.assert_allclose(a_2d.T, w._check_array(image=True))
        w.save_itx(OUTDIR / 'imgwave_2d_norm.itx')
        w.save_itx(OUTDIR / 'imgwave_2d_img.itx', image=True)
        w.save(OUTDIR / 'imgwave_2d_norm.ibw')
        w.save(OUTDIR / 'imgwave_2d_img.ibw', image=True)
        # 3d
        a_3d = a.reshape((2, 3, 4))
        w = IgorWave(a_3d, 'imagewave3d')
        np.testing.assert_allclose(a_3d, w._check_array(image=False))
        np.testing.assert_allclose(np.transpose(a_3d, (1, 0, 2)), w._check_array(image=True))
        w.save_itx(OUTDIR / 'imgwave_3d_norm.itx')
        w.save_itx(OUTDIR / 'imgwave_3d_img.itx', image=True)
        w.save(OUTDIR / 'imgwave_3d_norm.ibw')
        w.save(OUTDIR / 'imgwave_3d_img.ibw', image=True)
        # 4d
        a_4d = a.reshape((1, 2, 3, 4))
        w = IgorWave(a_4d, 'imagewave4d')
        np.testing.assert_allclose(a_4d, w._check_array(image=False))
        np.testing.assert_allclose(np.transpose(a_4d, (1, 0, 2, 3)), w._check_array(image=True))
        w.save_itx(OUTDIR / 'imgwave_4d_norm.itx')
        w.save_itx(OUTDIR / 'imgwave_4d_img.itx', image=True)
        w.save(OUTDIR / 'imgwave_4d_norm.ibw')
        w.save(OUTDIR / 'imgwave_4d_img.ibw', image=True)

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
            self.assertEqual(w._dimension_labels[0][-1].rstrip(b'\x00'), b'myindex')
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

    def test_long_name(self):
        long_name = 'LongWaveNameTest_NameLongerThan32Bytes'
        wave = IgorWave([1.0, 2.0, 3.0, 4.0, 5.0], name=long_name, long_name_support=True)
        self.assertEqual(wave._wave_header.bname, b'')
        self.assertEqual(wave.name, long_name)
        with open(OUTDIR / 'long_name.ibw', 'w+b') as fp:
            wave.save(fp)
            fp.seek(0)
            content = fp.read()
            self.assertIn(long_name.encode('utf-8'), content)
        with open(OUTDIR / 'long_name.itx', 'w+t') as fp:
            wave.save_itx(fp)
            fp.seek(0)
            content = fp.read()
            self.assertRegex(content, f"WAVES.*'{long_name}'")
        # with dimension labels
        wave.set_dimlabel(0, 0, 'label0')
        wave.set_dimlabel(0, 2, 'label2')
        wave.set_dimlabel(0, 3, 'X'*33)
        self.assertEqual(wave._bin_header.dimLabelsSize[0], 1+7+1+7+34)
        with open(OUTDIR / 'long_name_and_dimlabels.ibw', 'w+b') as fp:
            wave.save(fp)
            fp.seek(0)
            content = fp.read()
            self.assertIn(b'label0\x00\x00label2\x00', content)
        with open(OUTDIR / 'long_name_and_dimlabels.itx', 'w+t') as fp:
            wave.save_itx(fp)


if __name__ == '__main__':
    unittest.main()
