IgorWriter
==========

Write IGOR binary (.ibw) or text (.itx) files from numpy array

Usage
-----
    >>> import numpy as np
    >>> from igorwriter import IgorWave
    >>> array = np.array([[1,2,3],[4,5,6]])
    >>> wave = IgorWave(array, name='mywave')
    >>> wave.save('mywave.ibw')
    >>> wave.save_itx('mywave.itx')

Notes on Image Plots
--------------------
Image Plot in IGOR and :code:`imshow` in matplotlib use different convention for x and y axes:

- Rows as x, columns as y (IGOR)
- Columns as x, rows as y (Matplotlib)

Thus, :code:`image` parameter was introduced in :code:`save()` and :code:`save_itx()` methods. 
If you use e.g. 

    >>> wave.save('path.ibw', image=True)
    
:code:`plt.imshow` and Image Plot will give the same results.
