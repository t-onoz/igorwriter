IgorWriter
==========

Write IGOR binary (.ibw) or text (.itx) files from numpy array

Installation
------------
.. code-block:: doscon

    $ pip install igorwriter

Usage
-----
>>> import numpy as np
>>> from igorwriter import IgorWave
>>> # 64-bit integer is not supported in this package. Use np.int32, which is the default integer type for win64
>>> array = np.array([1,2,3,4,5,6], dtype=np.int32)
>>> wave = IgorWave(array, name='mywave')
>>> wave2 = IgorWave(array.astype(np.float32), name='mywave2')
>>> wave.set_datascale('DataUnit')
>>> wave.set_dimscale('x', 0, 0.01, 's')  # set x scale information
>>> wave.save('mywave.ibw')  # Igor Binary Wave files can only contain one wave per file
>>> wave.save_itx('mywave.itx')
>>> with open('multi_waves.itx', 'w') as fp:  # Igor Text files can contain multiples waves per file
>>>     wave.save_itx(fp)
>>>     wave2.save_itx(fp)

Wave Names
----------
There are restrictions on object names in IGOR. From v0.2.0, this package deals with illegal object names.

>>> wave = IgorWave(array, name='\'this_is_illegal\'', on_errors='fix')  # fix illegal names automatically
RenameWarning: name "'this_is_illegal'" is fixed as '_this_is_illegal_' (reason: name must not contain " ' : ; or any control characters.)
>>> print(wave.name)
_this_is_illegal_
>>> wave = IgorWave(array, name='\'this_is_illegal\'', on_errors='raise')  # raise errors
Traceback (most recent call last):
...
igorwriter.validator.InvalidNameError: name must not contain " ' : ; or any control characters.

Exporting pandas.DataFrame
--------------------------
Convenience functions for saving DataFrame in a Igor Text file or a series of Igor Binary Wave files are provided.

>>> from igorwriter import utils
>>> utils.dataframe_to_itx(df, 'df.itx')   # all Series are exported in one file
>>> waves = utils.dataframe_to_ibw(df, prefix='df_bin')   # each Series is saved in a separate file, <prefix>_<column>.ibw
>>> waves  # dictionary of generated IgorWaves. You can change wave names, set data units, set dimension scaling, etc.
{'col1': <IgorWave 'col1' at 0x...>, 'col2': ...}

Notes on Image Plots
--------------------
Image Plot in IGOR and :code:`imshow` in matplotlib use different convention for x and y axes:

- Rows as x, columns as y (IGOR)
- Columns as x, rows as y (Matplotlib)

Thus, :code:`image` parameter was introduced in :code:`save()` and :code:`save_itx()` methods. 
If you use e.g. 

>>> wave.save('path.ibw', image=True)
    
:code:`plt.imshow` and Image Plot will give the same results.