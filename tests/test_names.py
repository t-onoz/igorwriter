import warnings

import igorwriter.errors

try:
    import unittest2 as unittest
except ImportError:
    import unittest
from itertools import product

import igorwriter
from igorwriter import validator as v

encoding = "utf-8"

class NameTestCase(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', category=igorwriter.errors.RenameWarning)

    def test_empty_name(self):
        # empty string is invalid regardless of the modes.
        name = ''
        for liberal, long in product((True, False), (True, False)):
            with self.subTest(name=name, liberal=liberal, long=long):
                self.assertRaises(igorwriter.errors.InvalidNameError, v.check_and_encode, name, liberal, long)
                bname = v.check_and_encode(name, liberal, long, on_errors='fix')
                self.assertEqual(bname, b'wave0')  # empty name is fixed as 'wave0'

    def test_long_names(self):
        for liberal, long in product((True, False), (True, False)):
            with self.subTest('32-bit length name', liberal=liberal, long=long):
                name = 'x' * 32
                if long:
                    self.assertEqual(v.check_and_encode(name, liberal, long), name.encode(encoding))
                else:
                    self.assertRaises(igorwriter.errors.InvalidNameError, v.check_and_encode, name, liberal, long)
                    bname = v.check_and_encode(name, liberal, long, on_errors='fix')
                    desired = b'x'*31
                    self.assertEqual(bname, desired)
            with self.subTest('256-bit length name', liberal=liberal, long=long):
                name = 'x' * 256
                self.assertRaises(igorwriter.errors.InvalidNameError, v.check_and_encode, name, liberal, long)
                bname = v.check_and_encode(name, liberal, long, on_errors='fix')
                desired = b'x'*255 if long else b'x'*31
                self.assertEqual(bname, desired)

    def test_ng_letters(self):
        NG_LETTERS = ['"', '\'', ':', ';'] + [chr(i) for i in range(32)]
        for ng_letter, liberal, long in product(NG_LETTERS, (True, False), (True, False)):
            with self.subTest('with NG letter %r' % ng_letter, liberal=liberal, long=long):
                name = 'wave_' + ng_letter
                self.assertRaises(igorwriter.errors.InvalidNameError, v.check_and_encode, name, liberal, long)
                bname = v.check_and_encode(name, liberal, long, on_errors='fix')
                desired = b'wave__'
                self.assertEqual(bname, desired)

    def test_liberal_names(self):
        names = ('wave 0', '0 wave', 'Wave0-1', ' '*31, '0'*31)
        desired_std_short = (b'wave_0', b'X0_wave', b'Wave0_1', b'X'+b'_'*30, b'X'+b'0'*30)
        desired_std_long = (b'wave_0', b'X0_wave', b'Wave0_1', b'X'+b'_'*31, b'X'+b'0'*31)
        for (name, desired_s, desired_l), liberal, long in product(zip(names, desired_std_short, desired_std_long), (True, False), (True, False)):
            if liberal:
                bname = v.check_and_encode(name, liberal, long)
                self.assertEqual(bname, name.encode(encoding))
            else:
                self.assertRaises(igorwriter.errors.InvalidNameError, v.check_and_encode, name, liberal, long)
                bname = v.check_and_encode(name, liberal, long, on_errors='fix')
                desired = desired_l if long else desired_s
                self.assertEqual(bname, desired)

    def test_conflicts(self):
        names = (
            'append', 'APPEND',  # operations
            'abs', 'ABS',  # functions
            'do', 'DO',  # keywords
            'k1', 'K1',  # variables
        )
        for name, liberal, long in product(names, (True, False), (True, False)):
            self.assertRaises(igorwriter.errors.InvalidNameError, v.check_and_encode, name, liberal, long)
            bname = v.check_and_encode(name, liberal, long, on_errors='fix')
            desired = (name + '_').encode(encoding)
            self.assertEqual(bname, desired)


if __name__ == '__main__':
    unittest.main()
