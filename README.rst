IgorWriter
==========

Write IGOR binary (.ibw) or text (.itx) files from numpy array

Install
-------

Using pip::
    $ pip install .

Usage
-----
    import numpy as np
    from igorwriter import IgorWave
    array = np.array([[1,2,3],[4,5,6]])
    wave = IgorWave(array, name='mywave')
    wave.save('mywave.ibw')
    wave.save_itx('mywave.itx')

Notes on Image Plots
--------------------

When viewing 2d-array as an image, IGOR PRO uses row and column numbers as x and y values, respectively.
This is in contrast to `imshow` in matplotlib, which interpretes rows as y and columns as x.
`image` parameter was introduced in `save()` and `save_itx()` methods.
If you use e.g. `wave.save('path.ibw', image=True)`,  `plt.imshow` and Image Plot in IGOR will give the same results.
