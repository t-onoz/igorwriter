# -*- coding: utf-8 -*-
import os
import numpy as np

from igorwriter import IgorWave5


def demo():
    os.makedirs('./igor', exist_ok=True)
    a = np.arange(2*3*4*5)
    for typ in (int, float, complex):
        a_ = a.astype(typ)
        w1 = IgorWave5(a_, '%s_1d' % typ.__name__)
        w2 = IgorWave5(a_.reshape((2, -1)), '%s_2d' % typ.__name__)
        w3 = IgorWave5(a_.reshape((2, 3, -1)), '%s_3d' % typ.__name__)
        w4 = IgorWave5(a_.reshape((2, 3, 4, -1)), '%s_4d' % typ.__name__)
        with open('./igor/waves_%s.itx' % typ.__name__, 'w') as fp:
            # Igor Text files can contain multiple waves in one file.
            w1.save_itx(fp)
            w2.save_itx(fp)
            w3.save_itx(fp)
            w4.save_itx(fp)
        # Igor Binary Wave files can only contain one wave per file.
        w1.save('./igor/wave1_%s.ibw' % typ.__name__)
        w2.save('./igor/wave2_%s.ibw' % typ.__name__)
        w3.save('./igor/wave3_%s.ibw' % typ.__name__)
        w4.save('./igor/wave4_%s.ibw' % typ.__name__)


if __name__ == '__main__':
    demo()
