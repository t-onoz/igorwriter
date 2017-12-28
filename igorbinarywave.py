#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ctypes
import struct
from typing import Optional, BinaryIO, Union
import numpy as np

MAXDIMS = 4
MAX_WAVE_NAME2 = 18  # Maximum length of wave name in version 1 and 2 files. Does not include the trailing null.
MAX_WAVE_NAME5 = 31  # Maximum length of wave name in version 5 files. Does not include the trailing null.
MAX_UNIT_CHARS = 3

TYPES = {
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
        ('sIndicesSize', ctypes.c_int32),					# The size of string indicies if this is a text wave.
        ('optionsSize1', ctypes.c_int32),					# Reserved. Write zero. Ignore on read.
        ('optionsSize2', ctypes.c_int32),					# Reserved. Write zero. Ignore on read.
    ]

    def __init__(self):
        super().__init__()
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
        super().__init__()
        self.sfA = (1,) * MAXDIMS


class IgorBinaryWave(object):
    def __init__(self, array: Optional[np.ndarray]=None, name='wave0'):
        self._bin_header = BinHeader5()
        self._wave_header = WaveHeader5()
        self.array = array
        self.rename(name)
        self._extended_data_units = b''
        self._extended_dimension_units = [b'', b'', b'', b'']
    
    def rename(self, name: str):
        self._wave_header.bname = name.encode('ascii', errors='replace')

    def set_dimscale(self, dim, num1, num2, units: Optional[str]=None, flag='per_point'):
        dimint = {'x': 0, 'y': 1, 'z': 2, 't': 3}[dim]
        if flag == 'per_point':
            self._wave_header.sfB[dimint] = num1
            self._wave_header.sfA[dimint] = num2
        else:  # 'inclusive' scaling
            raise NotImplementedError('Only per_point scaling is supported.')
        if units is not None:
            bunits = units.encode('ascii', errors='replace')
            if len(bunits) <= 3:
                self._wave_header.dimUnits[dimint][:] = bunits + b'\x00' * (MAX_UNIT_CHARS + 1 - len(bunits))
                self._bin_header.dimEUnitsSize[dimint] = 0
                self._extended_dimension_units[dimint] = b''
            else:
                self._wave_header.dimUnits[dimint][:] = b'\x00' * (MAX_UNIT_CHARS + 1)
                self._bin_header.dimEUnitsSize[dimint] = len(bunits)
                self._extended_dimension_units[dimint] = bunits

    def set_datascale(self, units: str):
        bunits = units.encode('ascii', errors='replace')
        if len(bunits) <= 3:
            self._wave_header.dataUnits = bunits
            self._bin_header.dataEUnitsSize = 0
            self._extended_data_units = b''
        else:
            self._wave_header.dataUnits = b''
            self._bin_header.dataEUnitsSize = len(bunits)
            self._extended_data_units = bunits

    def save(self, file: Union[BinaryIO, str], image=False):
        if not isinstance(self.array, np.ndarray):
            raise ValueError('Please set an array before save')
        if self.array.ndim > 4:
            raise ValueError('Dimension of more than 4 is not supported.')
        if self.array.dtype.type is np.float16:
            # half-precision is not supported by IGOR, so convert to single-precision
            self.array = self.array.astype(np.float32)

        self._wave_header.npnts = len(self.array.ravel())
        self._wave_header.type = TYPES[self.array.dtype.type]

        if image and self.array.ndim >= 2:
            # transpose row and column
            a = np.transpose(self.array, (1, 0) + tuple(range(2, self.array.ndim)))
        else:
            a = self.array
        self._wave_header.nDim = a.shape + (0,) * (MAXDIMS - a.ndim)

        self._bin_header.wfmSize = 320 + self.array.nbytes

        # checksum
        first384bytes = (bytes(self._bin_header) + bytes(self._wave_header))[:384]
        self._bin_header.checksum -= sum(struct.unpack('@192h', first384bytes))

        if not hasattr(file, 'write'):
            file = open(file, mode='wb')

        file.write(self._bin_header)
        file.write(self._wave_header)
        file.write(a.tobytes(order='F'))

        file.write(self._extended_data_units)
        for u in self._extended_dimension_units:
            file.write(u)

    @staticmethod
    def load(self, file):
        raise NotImplementedError
