IgorWriter
==========

Write Igor Binary Wave (.ibw) or Igor Text (.itx) files from numpy array

Features
--------
* Compatible with multi-dimensional arrays (up to 4 dimensions)
* Supported :code:`numpy` data types: uint, int, float, complex, bool, str, bytes, datetime64
* Data units (:code:`IgorWave.set_datascale`)
* Dimension scaling (:code:`IgorWave.set_dimscale`)
* Dimension labels (:code:`IgorWave.set_dimlabel`)

Installation
------------
.. code-block:: doscon

    $ pip install igorwriter

Usage
-----

Basic usage

>>> import numpy as np
>>> from igorwriter import IgorWave
>>> array = np.array([1,2,3,4,5,6])
>>> # make IgorWave objects
>>> wave = IgorWave(array, name='mywave')
>>> wave2 = IgorWave(array.astype(np.float32), name='mywave2')
>>> # save data
>>> wave.save('mywave.ibw')
>>> wave.save_itx('mywave.itx')
>>> with open('multi_waves.itx', 'w') as fp:
>>>     # Igor Text files can contain multiples waves per file
>>>     wave.save_itx(fp)
>>>     wave2.save_itx(fp)

Data units, dimension scaling

>>> wave.set_datascale('DataUnit')
>>> wave.set_dimscale('x', 0, 0.01, 's')

A two-dimensional array with dimension labels

>>> a2 = np.random.random((10, 3))
>>> wave = IgorWave(a2, name='wave2d')
>>> # you may set optional dimension labels
>>> wave.set_dimlabel(0, -1, 'points') # label for entire rows
>>> wave.set_dimlabel(1, -1, 'values') # label for entire columns
>>> wave.set_dimlabel(1, 0, 'ValueA')  # label for column 0
>>> wave.set_dimlabel(1, 1, 'ValueB')  # label for column 1
>>> wave.set_dimlabel(1, 2, 'ValueC')  # label for column 2
>>> wave.save('my2dwave.ibw')

Unicode support
---------------
From igorwriter 0.5.0, IgorWave stores texts with utf-8 encoding.
If you use Igor Pro 6 or older and want to use non-ascii characters, set :code:`unicode=False` when calling :code:`IgorWave()`.
It will fall back to system text encoding (Windows-1252, Shift JIS, etc.).

Wave Names
----------
There are restrictions on object names in IGOR. From v0.2.0, this package deals with illegal object names.

>>> wave = IgorWave(array, name='wave:0', on_errors='fix')  # fix illegal names automatically
RenameWarning: name 'wave:0' is fixed as 'wave_0' (reason: name must not contain " ' : ; or any control characters.)
>>> wave.name
'wave_0'
>>> wave = IgorWave(array, name='wave:0', on_errors='raise')  # raise errors
Traceback (most recent call last):
...
InvalidNameError: name must not contain " ' : ; or any control characters.


Pint integration
----------------
If `Pint <https://github.com/hgrecco/pint>`_ Quantity objects are passed to IgorWave, data units will be set automatically.

>>> import pint
>>> ureg = pint.UnitRegistry()
>>> w = IgorWave([3, 4] * ureg.meter / ureg.second, name='wave_with_units')
>>> w.save_itx('wave_with_units.itx')
...
X SetScale d,0,0,"m / s",'wave_with_units'
X SetScale /P x,0.0,1.0,"",'wave_with_units'



Pandas integration
------------------
You can easily export DataFrame objects with convenience functions.

>>> from igorwriter import utils
>>> utils.dataframe_to_itx(df, 'df.itx')   # all Series are exported in one file
>>> waves = utils.dataframe_to_ibw(df, prefix='df_bin')   # each Series is saved in a separate file, <prefix>_<column>.ibw
>>> waves  # dictionary of generated IgorWaves. You can change wave names, set data units, set dimension scaling, etc.
{'col1': <IgorWave 'col1' at 0x...>, 'col2': ...}

IgorWriter tries to convert index info on pandas.Series objects.

* If the index is evenly-spaced, wave dimension scaling is set accordingly.
* Index names are interpreted as the dimension labels.

Notes on Image Plots
--------------------
Image Plot in IGOR and :code:`imshow` in matplotlib use different convention for x and y axes:

- Rows as x, columns as y (IGOR)
- Columns as x, rows as y (Matplotlib)

Thus, :code:`image` parameter was introduced in :code:`save()` and :code:`save_itx()` methods.
If you use e.g. 

>>> wave.save('path.ibw', image=True)
    
:code:`plt.imshow` and Image Plot will give the same results.


Changelog
=========


v0.5.0 (2023-07-08)
-------------------
- UTF-8 as default encoding. You can instead use system text encoding by setting :code:`unicode=False` to IgorWave().
- Automatic conversion from pandas index to dimension scaling.
- Exporting 64-bit integer waves (requires Igor Pro 7 or later).
- BUG FIX: Igor Text files created from np.bool\_ arrays were broken.


v0.4.1 (2023-07-02)
-------------------
- Added support for np.str\_, np.bytes\_ arrays.
- Automatic type conversion for np.object\_ arrays.
- Added support for dimension scaling (:code:`IgorWave.set_simlabel`).


v0.3.0 (2019-11-16)
-------------------
- Added :code:`utils.dict_to_{ibw, itx}` 
- Set datascale automatically for pint Quantity object.
- Added support for np.datetime64 array.


v0.2.3 (2019-11-09)
-------------------
- Added support for 64-bit integers (by automatically casting onto 32-bit integers on save). 


v0.2.0 (2019-11-08)
-------------------
- Added utilities for pandas (:code:`utils.dataframe_to_{ibw, itx}` ).
- Added unittest scripts. 
- Added wave name validator. 
- BUG FIX: long (> 3 bytes) units for dimension scaling were ignored in
  save_itx() 
- IgorWriter now uses system locale encoding rather than ASCII (the default behavior of
  IGOR Pro prior to ver. 7.00) 
