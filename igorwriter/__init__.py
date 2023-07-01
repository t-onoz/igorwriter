# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals
import os
import locale
import ctypes
import struct
import numpy as np
try:
    from pint.errors import UnitStrippedWarning
except ImportError:
    class UnitStrippedWarning(UserWarning):
        pass
import warnings

from igorwriter import validator
from igorwriter.errors import TypeConversionWarning

MAXDIMS = 4
MAX_WAVE_NAME2 = 18  # Maximum length of wave name in version 1 and 2 files. Does not include the trailing null.
MAX_WAVE_NAME5 = 31  # Maximum length of wave name in version 5 files. Does not include the trailing null.
MAX_UNIT_CHARS = 3

ENCODING = locale.getpreferredencoding()

TYPES = {
    np.bytes_: 0,
    np.str_: 0,
    np.bool_: 8,
    np.int8: 8,
    np.int16: 0x10,
    np.int32: 0x20,
    np.uint8: 8 + 0x40,
    np.uint16: 0x10 + 0x40,
    np.uint32: 0x20 + 0x40,
    np.float32: 2,
    np.float64: 4,
    np.complex64: 2 + 1,
    np.complex128: 4 + 1,
}
ITX_TYPES = {
    np.bool_: '/B',
    np.int8: '/B',
    np.int16: '/W',
    np.int32: '/I',
    np.uint8: '/U/B',
    np.uint16: '/U/W',
    np.uint32: '/U/I',
    np.float32: '/S',
    np.float64: '/D',
    np.complex64: '/C/S',
    np.complex128: '/C/D',
    np.bytes_: '/T',
    np.str_: '/T',
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
        ('whpad1', ctypes.c_char * 6),  # Reserved. Write zero. Ignore on read.
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
        ('spare', ctypes.c_char * 3),

        ('whUnused', ctypes.c_int32 * 13),  # Reserved. Write zero. Ignore on read.

        ('vRefNum', ctypes.c_int32),
        ('dirID', ctypes.c_int32),  # Used in memory only. Write zero. Ignore on read.

        # The following stuff is considered private to Igor.
        ('private', ctypes.c_byte * 28),

        # ('wData', ctypes.c_float * 1),  # The start of the array of data. Must be 64 bit aligned.
    ]

    def __init__(self):
        super(WaveHeader5, self).__init__()
        self.sfA = (1,) * MAXDIMS


class IgorWave5(object):
    def __init__(self, array, name='wave0', on_errors='fix'):
        """

        :param array: array_like object
        :param name: wave name
        :param on_errors: behavior when invalid name is given. 'fix': fix errors. 'raise': raise exception.
        """
        self._bin_header = BinHeader5()
        self._wave_header = WaveHeader5()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UnitStrippedWarning)
            self.array = np.asarray(array)
        self.rename(name, on_errors=on_errors)
        self._extended_data_units = b''
        self._extended_dimension_units = [b'', b'', b'', b'']
        self._dimension_labels = [dict(), dict(), dict(), dict()]
        try:
            # set units for pint.quantity._Quantity
            self.set_datascale('{:~}'.format(array.units))
        except (ValueError, AttributeError):
            pass
    
    def rename(self, name, on_errors='fix'):
        """

        :param name: new wavename.
        :param on_errors: behavior when invalid name is given. 'fix': fix errors. 'raise': raise exception.
        :return:
        """
        bname = validator.check_and_encode(name, on_errors=on_errors)
        self._wave_header.bname = bname

    @property
    def name(self):
        return self._wave_header.bname.decode(ENCODING)

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
            bunits = units.encode(ENCODING)
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
        bunits = units.encode('ascii', errors='replace')
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
        :param label: label string (up to 31 characters)
        """
        if dimNumber not in [0, 1, 2, 3]:
            raise ValueError('dimNumber must be 0, 1, 2, or 3.')

        # Dimension labels cannot contain illegal characters (", ', :, ; and control characters),
        # but they can conflict with built-in names.
        for s in validator.NG_LETTERS:
            if s in label:
                raise validator.InvalidNameError('label contains illegal characters (", \', :, ;, and control characters)')
        blabel = label.encode(ENCODING)
        if len(blabel) > 31:
            raise ValueError('Dimension labels cannot be longer than 31 bytes.')

        if label == '':
            # if label is empty, remove the label.
            self._dimension_labels[dimNumber].pop(dimIndex, None)
        else:
            self._dimension_labels[dimNumber][dimIndex] = blabel

        # calculate new dimLabelsSize
        if self._dimension_labels[dimNumber]:
            # Each dimension has max(dimIndex) + 2 (-1, 0, .. max(dimIndex)) labels, and each of them contains 32 bytes
            dimLabelsSize = 32 * (max(self._dimension_labels[dimNumber]) + 2)
        else:
            dimLabelsSize = 0
        self._bin_header.dimLabelsSize[dimNumber] = dimLabelsSize

    def save(self, file, image=False):
        """save data as igor binary wave (.ibw) format.

        :param file: file name or binary-file object.
        :param image: if True, rows and columns are transposed."""
        a = self._check_array(image=image)

        self._wave_header.npnts = len(a.ravel())
        self._wave_header.type = TYPES[a.dtype.type]

        self._wave_header.nDim = a.shape + (0,) * (MAXDIMS - a.ndim)

        if TYPES[a.dtype.type] == 0:
            # text wave
            wavesize = len(b''.join(a.ravel(order='F')))
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
            fp.write(self._bin_header)
            fp.write(self._wave_header)
            if TYPES[a.dtype.type] == 0: # text waves
                fp.write(b''.join(a.ravel(order='F')))
            else:
                fp.write(a.tobytes(order='F'))

            fp.write(self._extended_data_units)
            for u in self._extended_dimension_units:
                fp.write(u)
            for dimlabeldict in self._dimension_labels:
                if dimlabeldict:
                    for i in range(-1, max(dimlabeldict)+1):
                        b = dimlabeldict.get(i, b'\x00')
                        b += b'\x00' * (32-len(b))
                        assert len(b) == 32
                        fp.write(b)
            if TYPES[a.dtype.type] == 0:  # text waves
                sindices = np.zeros(a.size, dtype=np.int32)
                pos = 0
                for idx, s in enumerate(a.ravel(order='F')):
                    pos += len(s)
                    sindices[idx] = pos
                fp.write(sindices.tobytes(order='F'))
        finally:
            if fp is not file:
                fp.close()

    def save_itx(self, file, image=False):
        """save data as igor text (.itx) format.

        :param file: file name or text-file object.
        :param image: if True, rows and columns are transposed."""
        array = self._check_array(image=image)
        name = self._wave_header.bname.decode()

        fp = file if hasattr(file, 'write') else open(file, mode='w')
        try:
            if fp.tell() == 0:
                fp.write('IGOR\n')
            fp.write("WAVES {typ} /N=({shape}) '{name}'\n".format(
                typ=ITX_TYPES[array.dtype.type],
                shape=','.join(str(x) for x in array.shape),
                name=name
            ))
            fp.write('BEGIN\n')
            if np.iscomplexobj(array):
                def str_(x):
                    return '%s\t%s' % (x.real, x.imag)
            elif ITX_TYPES[array.dtype.type] == '/T':
                def str_(x):
                    return '"' + self._escape_specials(x).decode(ENCODING) + '"'
            else:
                str_ = str
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
            fp.write('X SetScale d,0,0,"{units}",\'{name}\'\n'.format(
                units=(self._wave_header.dataUnits or self._extended_data_units).decode(),
                name=name
            ))
            for idx, dim in list(enumerate(('x', 'y', 'z', 't')))[:array.ndim]:
                dimUnits = self._wave_header.dimUnits[idx][:]
                bunits = dimUnits.replace(b'\x00', b'') or self._extended_dimension_units[idx]
                fp.write('X SetScale /P {dim},{start},{delta},"{units}",\'{name}\'\n'.format(
                    dim=dim, start=self._wave_header.sfB[idx], delta=self._wave_header.sfA[idx],
                    units=bunits.decode(ENCODING),
                    name=name
                ))
            for dimNumber, dimlabeldict in enumerate(self._dimension_labels):
                for dimIndex, blabel in dimlabeldict.items():
                    fp.write('X SetDimLabel {dimNumber},{dimIndex},\'{label}\',\'{name}\'\n'.format(
                        dimNumber=dimNumber, dimIndex=dimIndex, label=blabel.decode(ENCODING), name=name
                    ))
        finally:
            if fp is not file:
                fp.close()

    def _check_array(self, image=False):
        if not isinstance(self.array, np.ndarray):
            raise ValueError('Please set an array before save')
        a = self._cast_array()
        if a.dtype.type not in TYPES:
            raise TypeError('Unsupported dtype: %r' % a.dtype.type)
        if a.ndim > 4:
            raise ValueError('Dimension of more than 4 is not supported.')

        if image and a.ndim >= 2:
            # transpose row and column
            a = np.transpose(a, (1, 0) + tuple(range(2, a.ndim)))
        return a

    def _cast_array(self):
        # check array dtype and try type casting if necessary
        type_ = self.array.dtype.type
        if type_ is np.float16:
            return self.array.astype(np.float32)
        if type_ is np.datetime64:
            self.set_datascale('dat')
            return (self.array - np.datetime64('1904-01-01 00:00:00')) / np.timedelta64(1, 's')
        if type_ is np.str_:
            return np.array([e.encode(ENCODING) for e in self.array.ravel()]).reshape(self.array.shape)
        for from_, to_ in {np.int64: np.int32, np.uint64: np.uint32}.items():
            if type_ is from_:
                type_info = np.iinfo(to_)
                if np.all((self.array >= type_info.min) & (self.array <= type_info.max)):
                    return self.array.astype(to_)
                else:
                    raise TypeError('Cast from %r to %r failed.' % (type_, to_))
        if type_ is np.object_:
            # infer data type
            candidates = [np.float64, np.bytes_, np.str_]
            for t in candidates:
                try:
                    a = self.array.astype(t)
                    msg = ("Data will be converted from np.object_ to numpy.{}. "
                           "To avoid this warning, "
                           "you may manually convert the data before calling IgorWave().").format(t.__name__)
                    if t is np.str_:
                        a = np.array([e.encode(ENCODING) for e in a.ravel()]).reshape(a.shape)
                    warnings.warn(msg, category=TypeConversionWarning)
                    return a
                except Exception:
                    pass
        return self.array

    @staticmethod
    def _escape_specials(b: bytes):
        # escape special characters
        b = b.replace(b'\\', b'\\\\')
        b = b.replace(b'\t', b'\\t')
        b = b.replace(b'\r', b'\\r')
        b = b.replace(b'\n', b'\\n')
        b = b.replace(b'\'', b'\\\'')
        b = b.replace(b'"', b'\\"')
        return b

    @staticmethod
    def load(self, file):
        raise NotImplementedError

    def __repr__(self):
        return '<IgorWave \'%s\' at %s>' % (self.name, hex(id(self)))


IgorWave = IgorWave5
