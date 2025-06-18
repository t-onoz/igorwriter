import os
import ctypes
import struct
import warnings
from locale import getpreferredencoding as _getpreferredencoding

import numpy as np
try:
    from pandas import Series
except ImportError:
    class Series:
        pass
try:
    from pint import Quantity
except ImportError:
    class Quantity:
        pass

import igorwriter.errors
from igorwriter import validator
from igorwriter.errors import TypeConversionWarning, StrippedSeriesIndexWarning, InvalidNameError

MAXDIMS = 4
MAX_WAVE_NAME2 = 18  # Maximum length of wave name in version 1 and 2 files. Does not include the trailing null.
MAX_WAVE_NAME5 = 31  # Maximum length of wave name in version 5 files. Does not include the trailing null.
MAX_UNIT_CHARS = 3
ENCODING = None

TYPES = {
    np.bytes_: 0,
    np.int8: 8,
    np.int16: 0x10,
    np.int32: 0x20,
    np.int64: 0x80,  # requires Igor Pro 7 or later
    np.uint8: 8 + 0x40,
    np.uint16: 0x10 + 0x40,
    np.uint32: 0x20 + 0x40,
    np.uint64: 0x80 + 0x40,  # requires Igor Pro 7 or later
    np.float32: 2,
    np.float64: 4,
    np.complex64: 2 + 1,
    np.complex128: 4 + 1,
}
ITX_TYPES = {
    np.int8: '/B',
    np.int16: '/W',
    np.int32: '/I',
    np.int64: '/L',  # requires Igor Pro 7 or later
    np.uint8: '/U/B',
    np.uint16: '/U/W',
    np.uint32: '/U/I',
    np.uint64: '/U/L',   # requires Igor Pro 7 or later
    np.float32: '/S',
    np.float64: '/D',
    np.complex64: '/C/S',
    np.complex128: '/C/D',
    np.bytes_: '/T',
}


class BinHeader5(ctypes.Structure):
    _pack_ = 2
    _fields_ = [
        ('version', ctypes.c_int16),					# Version number for backwards compatibility.
        ('checksum', ctypes.c_int16),				    # Checksum over this header and the wave header.
        ('wfmSize', ctypes.c_int32),						# The size of the WaveHeader5 data structure plus the wave data.
        ('formulaSize', ctypes.c_int32),					# The size of the dependency formula, including the null terminator, if any. Zero if no dependency formula.
        ('noteSize', ctypes.c_int32),						# The size of the note text.
        ('dataEUnitsSize', ctypes.c_int32),				    # The size of optional extended data units.
        ('dimEUnitsSize', ctypes.c_int32 * MAXDIMS),		# The size of optional extended dimension units.
        ('dimLabelsSize', ctypes.c_int32 * MAXDIMS),		# The size of optional dimension labels.
        ('sIndicesSize', ctypes.c_int32),					# The size of string indices if this is a text wave.
        ('optionsSize1', ctypes.c_int32),					# Reserved. Write zero. Ignore on read.
        ('optionsSize2', ctypes.c_int32),					# Reserved. Write zero. Ignore on read.
    ]

    def __init__(self):
        super(BinHeader5, self).__init__()
        self.version = 5


class WaveHeader5(ctypes.Structure):
    _pack_ = 2
    _fields_ = [
        ('next', ctypes.c_byte * 4),  # link to next wave in linked list.
        ('creationDate', ctypes.c_uint32),  # DateTime of creation.
        ('modDate', ctypes.c_uint32),  # DateTime of last modification.
        ('npnts', ctypes.c_int32),  # Total number of points (multiply dimensions up to first zero).
        ('type', ctypes.c_int16),  # See types (e.g. NT_FP64) above. Zero for text waves.
        ('dLock', ctypes.c_int16),  # Reserved. Write zero. Ignore on read.
        ('whpad1', ctypes.c_char * 3),  # Reserved. Write zero. Ignore on read.
        ('waveNameEncoding', ctypes.c_byte),
        ('waveUnitsEncoding', ctypes.c_byte),
        ('waveNoteEncoding', ctypes.c_byte),
        ('whVersion', ctypes.c_int16),  # Write 1. Ignore on read.
        ('bname', ctypes.c_char * (MAX_WAVE_NAME5 + 1)),  # Name of wave plus trailing null.
        ('whpad2', ctypes.c_int32),  # Reserved. Write zero. Ignore on read.
        ('dFolder', ctypes.c_byte * 4),  # Used in memory only. Write zero. Ignore on read.

        # Dimensioning info. [0] == rows, [1] == cols etc
        ('nDim', ctypes.c_int32 * MAXDIMS),  # Number of items in a dimension -- 0 means no data.
        ('sfA', ctypes.c_double * MAXDIMS),  # Index value for element e of dimension d = sfA[d]*e + sfB[d].
        ('sfB', ctypes.c_double * MAXDIMS),

        # SI units
        ('dataUnits', ctypes.c_char * (MAX_UNIT_CHARS + 1)),  # Natural data units go here - null if none.
        ('dimUnits', (ctypes.c_char * (MAX_UNIT_CHARS + 1)) * MAXDIMS),  # Natural dimension units go here - null if none.

        ('fsValid', ctypes.c_int16),  # TRUE if full scale values have meaning.
        ('whpad3', ctypes.c_int16),  # Reserved. Write zero. Ignore on read.
        ('botFullScale', ctypes.c_double),
        ('topFullScale', ctypes.c_double),  # The max and max full scale value for wave.

        ('dataEUnits', ctypes.c_byte * 4),  # Used in memory only. Write zero. Ignore on read.
        ('dimEUnits', (ctypes.c_byte * 4) * MAXDIMS),  # Used in memory only. Write zero. Ignore on read.
        ('dimLabels', (ctypes.c_byte * 4) * MAXDIMS),  # Used in memory only. Write zero. Ignore on read.
        
        ('waveNoteH', ctypes.c_byte * 4),  # Used in memory only. Write zero. Ignore on read.

        ('platform', ctypes.c_char),  # 0=unspecified, 1=Macintosh, '2=Windows', Added for Igor Pro 5.5.
        ('spare', ctypes.c_char),
        ('waveDimLabelEncoding', ctypes.c_byte),
        ('textWaveContentEncoding', ctypes.c_byte),

        ('whUnused', ctypes.c_int32 * 13),  # Reserved. Write zero. Ignore on read.

        ('vRefNum', ctypes.c_int32),
        ('dirID', ctypes.c_int32),  # Used in memory only. Write zero. Ignore on read.

        # The following stuff is considered private to Igor.
        ('private', ctypes.c_byte * 28),

        # ('wData', ctypes.c_float * 1),  # The start of the array of data. Must be 64 bit aligned.
    ]

    def __init__(self):
        super(WaveHeader5, self).__init__()
        self.sfA[:] = (1,) * MAXDIMS
        self.whVersion = 1


class IgorWave5(object):
    def __init__(self, array, name='wave0', on_errors='fix', unicode=True, int64_support=False):
        """

        :param array: array_like object
        :param name: wave name
        :param on_errors: behavior when invalid name is given. 'fix': fix errors. 'raise': raise exception.
        :param unicode: enables unicode support (encoding texts with utf-8).
            If you use Igor Pro 6 or older and want to use non-ascii characters, set it to False.
        :param int64_support: enables 64-bit integer support (requires Igor Pro 7 or later).
            Note: If enabled, it will break backward compatibility for Igor Pro 6 or earlier.
        """
        self._bin_header = BinHeader5()
        self._wave_header = WaveHeader5()
        if unicode:
            self._wave_header.waveNameEncoding = 1
            self._wave_header.waveUnitsEncoding = 1
            self._wave_header.waveNoteEncoding = 1
            self._wave_header.waveDimLabelEncoding = 1
            self._wave_header.textWaveContentEncoding = 1
            self._encoding = 'utf-8'
        else:
            self._encoding = _getpreferredencoding()
        self._int64_support = int64_support
        self.rename(name, on_errors)
        self._note = b''
        self._formula = b''
        self._extended_data_units = b''
        self._extended_dimension_units = [b'', b'', b'', b'']
        self._dimension_labels = (dict(), dict(), dict(), dict())

        if isinstance(array, Series):
            self.array = self._parse_Series(array)
        elif isinstance(array, Quantity):
            self.array = self._parse_Pint(array)
        else:
            self.array = np.asarray(array)

    def rename(self, name, on_errors='fix'):
        """

        :param name: new wavename.
        :param on_errors: behavior when invalid name is given. 'fix': fix errors. 'raise': raise exception.
        :return:
        """
        bname = validator.check_and_encode(name, on_errors=on_errors, encoding=self._encoding)
        self._wave_header.bname = bname

    @property
    def name(self):
        return self._wave_header.bname.decode(self._encoding)

    def set_dimscale(self, dim, start, delta, units=None):
        """Set scale information of each axis.

        :param dim: dimensionality, 'x', 'y', 'z', or 't'
        :param start: start value (e.g. x[0])
        :param delta: "delta value" (e.g. x[1] - x[0])
        :param units: optional unit string
        :return:
        """
        dimint = {'x': 0, 'y': 1, 'z': 2, 't': 3}[dim]
        self._wave_header.sfB[dimint] = start
        self._wave_header.sfA[dimint] = delta
        if units is not None:
            bunits = units.encode(self._encoding)
            # if the units is short, they are written directly in the WaveHeader5 object.
            # longer units are stored as dict, and encoded when saved.
            if len(bunits) <= MAX_UNIT_CHARS:
                self._wave_header.dimUnits[dimint][:] = bunits + b'\x00' * (MAX_UNIT_CHARS + 1 - len(bunits))
                self._bin_header.dimEUnitsSize[dimint] = 0
                self._extended_dimension_units[dimint] = b''
            else:
                self._wave_header.dimUnits[dimint][:] = b'\x00' * (MAX_UNIT_CHARS + 1)
                self._bin_header.dimEUnitsSize[dimint] = len(bunits)
                self._extended_dimension_units[dimint] = bunits

    def set_datascale(self, units):
        """Set units of the data.

        :param units: string representing units of the data.
        """
        bunits = units.encode(self._encoding)
        # if the units is short, they are written directly in the WaveHeader5 object.
        # longer units are stored as python bytes object, and written when saved.
        if len(bunits) <= MAX_UNIT_CHARS:
            self._wave_header.dataUnits = bunits
            self._bin_header.dataEUnitsSize = 0
            self._extended_data_units = b''
        else:
            self._wave_header.dataUnits = b''
            self._bin_header.dataEUnitsSize = len(bunits)
            self._extended_data_units = bunits

    def set_dimlabel(self, dimNumber: int, dimIndex: int, label: str):
        """Set dimension labels.

        :param dimNumber: 0 (rows), 1 (columns), 2 (layers), or 3 (chunks)
        :param dimIndex: if -1, sets the label for the entire dimension, if >= 0, sets the label for that element of the dimension.
        :param label: label string (up to 31 characters). Clears the label if empty ('').
        """
        if dimNumber not in [0, 1, 2, 3]:
            raise ValueError('dimNumber must be 0, 1, 2, or 3.')

        if label:
            blabel = validator.check_and_encode(label, liberal=True, long=False, allow_builtins=True)
            self._dimension_labels[dimNumber][dimIndex] = blabel + b'\x00' * (32 - len(blabel))
        else:
            self._dimension_labels[dimNumber].pop(dimIndex, None)

        # calculate new dimLabelsSize
        if self._dimension_labels[dimNumber]:
            # Each dimension has max(dimIndex) + 2 (-1, 0, .. max(dimIndex)) labels, and each of them contains 32 bytes
            dimLabelsSize = 32 * (max(self._dimension_labels[dimNumber]) + 2)
        else:
            dimLabelsSize = 0
        self._bin_header.dimLabelsSize[dimNumber] = dimLabelsSize

    def set_note(self, note):
        """Set the wave note.

        :param note: a string that represents the wave note. Clears the note if empty (note='').
        """
        self._note = note.encode(self._encoding)
        self._bin_header.noteSize = len(self._note)

    def set_formula(self, formula):
        """set wave dependency formula.

        :param formula: a string that represents the dependency formula. Clears the formula if empty (formula='')."""
        bformula = formula.encode(self._encoding) + b'\x00' if formula else b''
        self._formula = bformula
        self._bin_header.formulaSize = len(bformula)

    set_wavenote = set_note

    def save(self, file, image=False):
        """save data as igor binary wave (.ibw) format.

        :param file: file name or binary-file object.
        :param image: if True, rows and columns are transposed."""
        a = self._check_array(image=image)
        self._array_saved = a

        self._wave_header.npnts = len(a.ravel())
        self._wave_header.type = TYPES[a.dtype.type]

        self._wave_header.nDim = a.shape + (0,) * (MAXDIMS - a.ndim)

        if TYPES[a.dtype.type] == 0:
            # text wave (should be stored as numpy.bytes_)
            wavesize = np.sum([len(x) for x in a.ravel()], dtype=int)
            self._bin_header.sIndicesSize = 4 * a.size
        else:
            wavesize = a.nbytes
            self._bin_header.sIndicesSize = 0

        self._bin_header.wfmSize = 320 + wavesize

        # checksum
        first384bytes = (bytearray(self._bin_header) + bytearray(self._wave_header))[:384]
        self._bin_header.checksum -= sum(struct.unpack('@192h', first384bytes))

        fp = file if hasattr(file, 'write') else open(file, mode='wb')
        fp.seek(0, os.SEEK_END)
        if fp.tell() > 0:
            raise ValueError('You can only save() into an empty file.')
        try:
            # the binary header and wave header
            fp.write(self._bin_header)
            fp.write(self._wave_header)

            # wave data
            if TYPES[a.dtype.type] == 0:  # text waves
                fp.write(b''.join(a.ravel(order='F')))
            else:
                fp.write(a.tobytes(order='F'))

            # dependency formula
            fp.write(self._formula)

            # wave note
            fp.write(self._note)

            # extended data units, dimension units
            fp.write(self._extended_data_units)
            for u in self._extended_dimension_units:
                fp.write(u)

            # dimension labels
            for dimlabeldict in self._dimension_labels:
                if dimlabeldict:
                    for i in range(-1, max(dimlabeldict)+1):
                        b = dimlabeldict.get(i, b'\x00'*32)
                        assert len(b) == 32
                        fp.write(b)

            # string indices if this is a text wave.
            if TYPES[a.dtype.type] == 0:
                sindices = np.zeros(a.size, dtype=np.int32)
                pos = 0
                for idx, s in enumerate(a.ravel(order='F')):
                    pos += len(s)
                    sindices[idx] = pos
                fp.write(sindices.tobytes())
        finally:
            if fp is not file:
                fp.close()

    def save_itx(self, file, image=False):
        """save data as igor text (.itx) format.

        :param file: file name or text-file object.
        :param image: if True, rows and columns are transposed."""
        array = self._check_array(image=image)
        name = self._wave_header.bname.decode(self._encoding)
        itx_type = ITX_TYPES[array.dtype.type]
        shape = ','.join(str(x) for x in array.shape)
        str_ = self._get_str_converter(array)

        fp = file if hasattr(file, 'write') else open(file, encoding=self._encoding, mode='w')
        try:
            if fp.tell() == 0:
                fp.write('IGOR\n')
            fp.write(f"WAVES {itx_type} /N=({shape}) '{name}'\n")
            fp.write('BEGIN\n')

            # write in column/row/layer/chunk order
            expanded = array
            while expanded.ndim < 4:
                expanded = np.expand_dims(expanded, expanded.ndim)
            for chunk in range(expanded.shape[3]):
                for layer in range(expanded.shape[2]):
                    for row in range(expanded.shape[0]):
                        fp.write('\t'.join(str_(x) for x in expanded[row, :, layer, chunk]))
                        fp.write('\n')
                    fp.write('\n')
                fp.write('\n')
            fp.write('END\n')

            # data units
            dataunits = (self._wave_header.dataUnits or self._extended_data_units).decode()
            fp.write(f'X SetScale d,0,0,"{dataunits}",\'{name}\'\n')

            # dimension scaling
            for idx, dim in list(enumerate(('x', 'y', 'z', 't')))[:array.ndim]:
                bdimUnits = (self._wave_header.dimUnits[idx][:].replace(b'\x00', b'') or self._extended_dimension_units[idx])
                dimUnits = bdimUnits.decode(self._encoding)
                start = self._wave_header.sfB[idx]
                delta = self._wave_header.sfA[idx]
                fp.write(f'X SetScale /P {dim},{start},{delta},"{dimUnits}",\'{name}\'\n')

            # dimension labels
            for dimNumber, dimlabeldict in enumerate(self._dimension_labels):
                for dimIndex, blabel in dimlabeldict.items():
                    label = blabel.rstrip(b'\x00').decode(self._encoding)
                    fp.write(f'X SetDimLabel {dimNumber},{dimIndex},\'{label}\',\'{name}\'\n')

            # dependency formula
            if self._formula:
                formula = self._formula.rstrip(b'\x00').decode(self._encoding)
                fp.write(f'X SetFormula \'{name}\', "{formula}"\n')

            # wave note
            if self._note:
                note = self._escape_specials(self._note.decode(self._encoding))
                fp.write(f'X Note \'{name}\', "{note}"\n')
        finally:
            if fp is not file:
                fp.close()

    def _get_str_converter(self, array):
        """get string converter based on the array data type. to be used in save_itx()."""
        if np.iscomplexobj(array):
            return lambda x: '%s\t%s' % (x.real, x.imag)
        elif array.dtype.type is np.bytes_:
            return lambda x: '"' + self._escape_specials(x.decode(self._encoding)) + '"'
        return str

    def _check_array(self, image=False):
        if not isinstance(self.array, np.ndarray):
            raise ValueError('Please set an array before save')
        a = self._cast_array()
        if a.ndim > 4:
            raise ValueError('Dimension of more than 4 is not supported.')

        if image and a.ndim >= 2:
            # transpose row and column
            a = np.transpose(a, (1, 0) + tuple(range(2, a.ndim)))
        return a

    def _cast_array(self):
        """
        Prepare the input array for IgorWave export:
        - Casts to compatible dtype (e.g., float32, int32, etc.)
        - Converts special types like datetime64 or str
        - Handles np.object_ via type inference
        - Modifies internal headers if necessary (e.g., for bytes or str types)

        Returns:
            np.ndarray: Converted array ready for export.

        Raises:
            OverflowError, TypeError, ValueError
        """
        type_ = self.array.dtype.type
        if type_ is np.bytes_:
            # 255 means binary data
            self._wave_header.textWaveContentEncoding = 255
            a = self.array
        elif (not self._int64_support) and type_ in (np.int64, np.uint64):
            # Convert to 32-bit integers
            to_type = {np.int64: np.int32, np.uint64: np.uint32}[type_]
            tinfo = np.iinfo(to_type)
            if (tinfo.min <= np.min(self.array)) and (np.max(self.array) <= tinfo.max):
                a = self.array.astype(to_type)
            else:
                msg = (f'Overflow detected when converting an array with type {type_!r}. '
                    'If you are using Igor Pro 7 or later, try setting int64_support=True when calling IgorWave().')
                raise OverflowError(msg)
        elif type_ is np.bool_:
            a = self.array.astype(np.int8)
        elif type_ is np.float16:
            a = self.array.astype(np.float32)
        elif type_ in (np.longdouble, getattr(np, 'float96', None), getattr(np, 'float128', None)):
            a = self.array.astype(np.float64)
        elif type_ in (np.clongdouble, getattr(np, 'complex96', None), getattr(np, 'complex256', None)):
            a = self.array.astype(np.complex128)
        elif type_ is np.datetime64:
            self.set_datascale('dat')
            a = (self.array - np.datetime64('1904-01-01 00:00:00')) / np.timedelta64(1, 's')
        elif type_ is np.str_:
            a = np.char.encode(self.array, encoding=self._encoding)
        elif type_ is np.object_:
            # infer data type
            candidates = [np.float64, np.str_, np.bytes_]
            for t in candidates:
                try:
                    a = self.array.astype(t)
                except (ValueError, UnicodeError):
                    continue
                else:
                    msg = ("Data will be converted from np.object_ to numpy.{}. "
                           "To avoid this warning, "
                           "you may manually convert the data before calling IgorWave().").format(t.__name__)
                    if t is np.str_:
                        a = np.char.encode(a, encoding=self._encoding)
                    elif t is np.bytes_:
                        # 255 means binary data
                        self._wave_header.textWaveContentEncoding = 255
                    warnings.warn(msg, category=TypeConversionWarning)
                    break
            else:
                raise ValueError('The array could not be converted to Igor-compatible types.')
        elif type_ in TYPES:
            a = self.array
        else:
            raise TypeError('The array data type %r is not compatible with Igor.' % type_)
        assert a.dtype.type in TYPES
        return a

    def _parse_Series(self, s):
        import pandas as pd
        start, step = None, None
        # parse pandas.Series objects
        if isinstance(s.index, pd.MultiIndex):
            msg = 'pandas MultiIndex is stripped because it is not compatible with Igor waves.'
            warnings.warn(msg, StrippedSeriesIndexWarning)
        elif isinstance(s.index, pd.RangeIndex):
            start, step = s.index.start, s.index.step
        else:
            i = s.index.to_numpy()
            if issubclass(i.dtype.type, np.number):
                diff = np.diff(i)
                if np.all(np.isclose(diff, diff[0], atol=0)):
                    start, step = i[0], diff[0]
                else:
                    msg = 'pandas Index is stripped because it is not evenly spaced.'
                    warnings.warn(msg, StrippedSeriesIndexWarning)
            else:
                msg = 'pandas Index is stripped because non-numeric scaling is not supported in Igor'
                warnings.warn(msg, StrippedSeriesIndexWarning)
        if start is not None:
            self.set_dimscale('x', start, step)
            if s.index.name:
                try:
                    self.set_dimlabel(0, -1, s.index.name)
                except (InvalidNameError, ValueError):
                    msg = 'index name is ignored because "{}" is not a valid dimension label.'.format(
                        s.index.name)
                    warnings.warn(msg)
        if issubclass(s.dtype.type, Quantity):
            self.set_datascale('{:~}'.format(s.pint.units))
            s = s.pint.magnitude
        return s.to_numpy()

    def _parse_Pint(self, q):
        # set units for pint.quantity._Quantity
        self.set_datascale('{:~}'.format(q.units))
        return q.magnitude

    @staticmethod
    def _escape_specials(s: str):
        # escape special characters
        s = s.replace('\\', '\\\\')
        s = s.replace('\t', '\\t')
        s = s.replace('\r', '\\r')
        s = s.replace('\n', '\\n')
        s = s.replace('\'', '\\\'')
        s = s.replace('"', '\\"')
        return s

    @staticmethod
    def load(self, file):
        raise NotImplementedError

    def __repr__(self):
        return '<IgorWave \'%s\' at %s>' % (self.name, hex(id(self)))


IgorWave = IgorWave5
