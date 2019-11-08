# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals
import locale
import ctypes
import struct
import numpy as np

from igorwriter import validator

MAXDIMS = 4
MAX_WAVE_NAME2 = 18  # Maximum length of wave name in version 1 and 2 files. Does not include the trailing null.
MAX_WAVE_NAME5 = 31  # Maximum length of wave name in version 5 files. Does not include the trailing null.
MAX_UNIT_CHARS = 3

ENCODING = locale.getpreferredencoding()

TYPES = {
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

        :param array: numpy.ndarray object
        :param name: wave name
        :param errors: behavior when invalid name is given. 'fix': fix errors. 'raise': raise exception.
        """
        self._bin_header = BinHeader5()
        self._wave_header = WaveHeader5()
        if array is None:
            self.array = np.array([], dtype=float)
        elif isinstance(array, np.ndarray):
            self.array = array
        else:
            self.array = np.array(array)
        self.rename(name, on_errors=on_errors)
        self._extended_data_units = b''
        self._extended_dimension_units = [b'', b'', b'', b'']
    
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

    def save(self, file, image=False):
        """save data as igor binary wave (.ibw) format.

        :param file: file name or binary-file object.
        :param image: if True, rows and columns are transposed."""
        a = self._check_array(image=image)

        self._wave_header.npnts = len(a.ravel())
        self._wave_header.type = TYPES[a.dtype.type]

        self._wave_header.nDim = a.shape + (0,) * (MAXDIMS - a.ndim)

        self._bin_header.wfmSize = 320 + a.nbytes

        # checksum
        first384bytes = (bytearray(self._bin_header) + bytearray(self._wave_header))[:384]
        self._bin_header.checksum -= sum(struct.unpack('@192h', first384bytes))

        fp = file if hasattr(file, 'write') else open(file, mode='wb')
        if fp.tell() > 0:
            raise ValueError('You can only save() into an empty file.')
        try:
            fp.write(self._bin_header)
            fp.write(self._wave_header)
            fp.write(a.tobytes(order='F'))

            fp.write(self._extended_data_units)
            for u in self._extended_dimension_units:
                fp.write(u)
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
        finally:
            if fp is not file:
                fp.close()

    def _check_array(self, image=False):
        if not isinstance(self.array, np.ndarray):
            raise ValueError('Please set an array before save')
        if self.array.dtype.type is np.float16:
            # half-precision is not supported by IGOR, so convert to single-precision
            a = self.array.astype(np.float32)
        else:
            a = self.array
        if self.array.dtype.type not in TYPES:
            raise TypeError('Unsupported dtype: %r' % self.array.dtype.type)
        if a.ndim > 4:
            raise ValueError('Dimension of more than 4 is not supported.')

        if image and a.ndim >= 2:
            # transpose row and column
            a = np.transpose(a, (1, 0) + tuple(range(2, a.ndim)))

        return a

    @staticmethod
    def load(self, file):
        raise NotImplementedError

    def __repr__(self):
        return '<IgorWave \'%s\' at %s>' % (self.name, hex(id(self)))


IgorWave = IgorWave5
