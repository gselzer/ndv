# test_example = true

from math import pi

import fastplotlib as fpl
import numpy as np
from fastplotlib.tools import HistogramLUTTool

figure = fpl.Figure(size=(700, 560))
# cmap_transform from an array, so the colors on the sine line will be based on the
# sine y-values
data = np.random.normal(10, 10, 10000).reshape((100, 100))
ig = fpl.ImageGraphic(data)
h = HistogramLUTTool(data, ig)
h.rotate(-pi / 2, "z")
g = figure[0, 0].add_graphic(h)
figure.show()


# NOTE: `if __name__ == "__main__"` is NOT how to use fastplotlib interactively
# please see our docs for using fastplotlib interactively in ipython and jupyter
if __name__ == "__main__":
    print(__doc__)
    fpl.run()
