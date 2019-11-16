from io import open

import igorwriter
from igorwriter import IgorWave


def dict_to_itx(dict_like, path_or_buf, on_errors='fix'):
    """

    :param dict_like: dictionary-like object ({name: array-like, ...})
    :param path_or_buf: filename or file-like object
    :param on_errors: behavior when invalid name is given. 'fix': fix errors. 'raise': raise an exception.
    :return: dictionary of generated waves ({name: wave, ...})
    """
    fp = path_or_buf if hasattr(path_or_buf, 'write') else open(path_or_buf, 'w', encoding=igorwriter.ENCODING)
    wavelist = [IgorWave(array, name, on_errors) for (name, array) in dict_like.items()]
    try:
        for wave in wavelist:
            wave.save_itx(fp)
    finally:
        if fp is not path_or_buf:
            fp.close()
    return {wave.name: wave for wave in wavelist}


def dict_to_ibw(dict_like, prefix, on_errors='fix'):
    """

    :param dict_like: dictionary-like object ({name: array-like, ...})
    :param prefix: file name prefix (each wave is saved as <prefix>_<column>.ibw)
    :param on_errors: behavior when invalid name is given. 'fix': fix errors. 'raise': raise an exception.
    :return: dictionary of generated waves ({name: wave, ...})
    """
    wavelist = [IgorWave(array, name, on_errors) for (name, array) in dict_like.items()]
    for wave in wavelist:
        wave.save(str(prefix) + '_%s.ibw' % wave.name)
    return {wave.name: wave for wave in wavelist}


dataframe_to_itx = dict_to_itx
dataframe_to_ibw = dict_to_ibw
