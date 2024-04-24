# IgorWriter

Write Igor Binary Wave (.ibw) or Igor Text (.itx) files from numpy array

## Features

-   Compatible with multi-dimensional arrays (up to 4 dimensions)
-   Supported `numpy` data types: uint, int, float, complex, bool, str,
    bytes, datetime64
-   Data units (`IgorWave.set_datascale`)
-   Dimension scaling (`IgorWave.set_dimscale`)
-   Dimension labels (`IgorWave.set_dimlabel`)
-   Wave notes (`IgorWave.set_note`)
-   Dependency formula (`IgorWave.set_formula`)

## Installation

``` doscon
$ pip install igorwriter
```

## Usage

Basic usage

```python
import numpy as np 
from igorwriter import IgorWave 

array = np.array([1,2,3,4,5,6]) 

# make IgorWave objects 
wave = IgorWave(array, name='mywave') 
wave2 = IgorWave(array.astype(np.float32), name='mywave2') 

# save data
wave.save('mywave.ibw') 
wave.save_itx('mywave.itx')

with open('multi_waves.itx', 'w') as fp: 
    # Igor Text files can contain multiples waves per file 
    wave.save_itx(fp)
    wave2.save_itx(fp)
```

Data units, dimension scaling

```python
wave.set_datascale('DataUnit') 
wave.set_dimscale('x', 0, 0.01, 's')
```

A two-dimensional array with dimension labels

```python
a2 = np.random.random((10, 3)) 
wave = IgorWave(a2, name='wave2d') 

# you may set optional dimension labels 
wave.set_dimlabel(0, -1, 'points') # label for entire rows 
wave.set_dimlabel(1, -1, 'values') # label for entire columns 
wave.set_dimlabel(1, 0, 'ValueA') # label for column 0 
wave.set_dimlabel(1, 1, 'ValueB') # label for column 1 
wave.set_dimlabel(1, 2, 'ValueC') # label for column 2 
wave.save('my2dwave.ibw')
```

You can append arbitrary Igor commands to Igor Text files.

```python
wave = IgorWave([1, 4, 9], name='wave0') 
with open('wave0.itx', 'w') as fp: 
    wave.save_itx(fp) 

fp.write("X Display 'wave0'\n")
```

## Unicode support

From igorwriter 0.5.0, IgorWave stores texts with utf-8 encoding. If you
use Igor Pro 6 or older and want to use non-ascii characters, set
`unicode=False` when calling `IgorWave()`. It will fall back to system
text encoding (Windows-1252, Shift JIS, etc.).

## Wave Names

There are restrictions on object names in IGOR. From v0.2.0, this
package deals with illegal object names.

```python
# This will issue a RenameWarning, and the name will be automatically changed.
wave = IgorWave(array, name='wave:0', on_errors='fix')
print(wave.name)  # wave_0

# This will raise an InvalidNameError.
wave = IgorWave(array, name='wave:0', on_errors='raise')
```

## Pint integration

If [Pint](https://github.com/hgrecco/pint) Quantity objects are passed
to IgorWave, data units will be set automatically.

```python
import pint 
ureg = pint.UnitRegistry() 
w = IgorWave([3, 4] * ureg.meter / ureg.second, name='wave_with_units')

w.save_itx('wave_with_units.itx')
print(open('wave_with_units.itx', 'r').read())
# ...
# X SetScale d,0,0,"m / s",'wave_with_units' 
# X SetScale /P x,0.0,1.0,"",'wave_with_units'
```

## Pandas integration

You can easily export DataFrame objects with convenience functions.

```python
from igorwriter import utils 
utils.dataframe_to_itx(df, 'df.itx') # all Series are exported in one file 
waves = utils.dataframe_to_ibw(df, prefix='df_bin') # each Series is saved in a separate file, <prefix>_<column>.ibw 
print(waves) # dictionary of generated IgorWaves
# {'col1': <IgorWave 'col1' at 0x...>, 'col2': ...}
```

IgorWriter tries to convert index info on `pandas.Series` objects in following manners.

-   If the index is evenly-spaced, wave dimension scaling is set
    accordingly.
-   Index names are interpreted as the dimension labels.

## Notes on Image Plots

Image Plot in IGOR and `imshow` in matplotlib use different convention
for x and y axes:

-   Rows as x, columns as y (IGOR)
-   Columns as x, rows as y (Matplotlib)

Thus, `image` parameter was introduced in `save()` and `save_itx()`
methods. If you use e.g.


wave.save('path.ibw', image=True)

`plt.imshow` and Image Plot will give the same results.

The `image=True` option transposes rows and columns of the underlying
array, but does not swap data units, dimension scaling, etc. (You need to
change them manually).

# Changelog

## v0.6.0 (2024-01-13)

-   Wave note support
-   Dependency formula support

## v0.5.0 (2023-07-08)

-   UTF-8 as default encoding. You can instead use system text encoding
    by setting `unicode=False` to IgorWave().
-   Automatic conversion from pandas index to dimension scaling.
-   Exporting 64-bit integer waves (requires Igor Pro 7 or later).
-   BUG FIX: Igor Text files created from `np.bool_` arrays were broken.

## v0.4.1 (2023-07-02)

-   Added support for `np.str_` and `np.bytes_` arrays.
-   Automatic type conversion for `np.object_` arrays.
-   Added support for dimension scaling (`IgorWave.set_simlabel`).

## v0.3.0 (2019-11-16)

-   Added `utils.dict_to_{ibw, itx}`
-   Set datascale automatically for pint Quantity object.
-   Added support for np.datetime64 array.

## v0.2.3 (2019-11-09)

-   Added support for 64-bit integers (by automatically casting onto
    32-bit integers on save).

## v0.2.0 (2019-11-08)

-   Added utilities for pandas (`utils.dataframe_to_{ibw, itx}` ).
-   Added unittest scripts.
-   Added wave name validator.
-   BUG FIX: long (\> 3 bytes) units for dimension scaling were ignored
    in save_itx()
-   IgorWriter now uses system locale encoding rather than ASCII (the
    default behavior of IGOR Pro prior to ver. 7.00)
