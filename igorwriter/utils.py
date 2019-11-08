from io import open

import igorwriter
from igorwriter import IgorWave


def dataframe_to_itx(dataframe, path_or_buf, on_errors='fix'):
    """

    :type dataframe: pandas.DataFrame
    :param path_or_buf: filename or file-like object
    :param on_errors: behavior when invalid name is given. 'fix': fix errors. 'raise': raise an exception.
    :return: dictionary of generated waves ({column: wave, ...}
    """
    waves = dict()
    fp = path_or_buf if hasattr(path_or_buf, 'write') else open(path_or_buf, 'w', encoding=igorwriter.ENCODING)
    try:
        for column, series in dataframe.items():
            wave = IgorWave(series.to_numpy(), column, on_errors=on_errors)
            waves[column] = wave
            wave.save_itx(fp)
    finally:
        if fp is not path_or_buf:
            fp.close()
    return waves


def dataframe_to_ibw(dataframe, prefix, on_errors='fix'):
    """

    :type dataframe: pandas.DataFrame
    :param prefix: file name prefix (each wave is saved as <prefix>_<column>.ibw)
    :param on_errors: behavior when invalid name is given. 'fix': fix errors. 'raise': raise an exception.
    :return: dictionary of generated waves ({column: wave, ...}
    """
    waves = dict()
    for column, series in dataframe.items():
        wave = IgorWave(series.to_numpy(), column, on_errors=on_errors)
        wave.save(str(prefix) + '_%s.ibw' % column)
        waves[column] = wave
    return waves
