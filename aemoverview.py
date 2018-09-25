import numpy as np

from multiprocessing import Pool

from scipy.interpolate import NearestNDInterpolator as interp

from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QGraphicsView
from PyQt5.QtWidgets import QGraphicsScene
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QGraphicsRectItem
from PyQt5.QtGui import QBrush
from PyQt5.QtGui import QPen
from PyQt5.QtCore import Qt

from PIL import Image
from PIL.ImageQt import ImageQt


class AEMOverview(QGraphicsScene):
    mouseMoved = pyqtSignal()

    def __init__(self, baseimg, trace, borehole,
                 tracewidth=1, boreholesize=4, tracecolor=(255,255,0), boreholecolor=(255,0,0)):
        super(AEMOverview, self).__init__()

        self.trace = trace

        pixelvalues = baseimg.load()

        self.division = np.zeros(baseimg.size, dtype=np.int)

        self.height = baseimg.size[1]

        self.tracewidth = tracewidth



        for z in trace:
            x = int(z[0])
            y = int(z[1])
            if (x >= 0 and x < baseimg.size[0] and y >= 0 and y < baseimg.size[1]):
                for i in range(x - tracewidth, x + tracewidth):
                    for j in range(y - tracewidth, y + tracewidth):
                        pixelvalues[i, j] = tracecolor
                for i in range(x - tracewidth*8, x + tracewidth*8):
                    for j in range(y - tracewidth*8, y + tracewidth*8):
                        self.division[i, j] = z[2]



        for z in borehole:
            x = z[0]
            y = z[1]
            if (x >= 0 and x < baseimg.size[0] and y >= 0 and y < baseimg.size[1]):
                for i in range(x - boreholesize, x + boreholesize):
                    for j in range(y - boreholesize, y + boreholesize):
                        pixelvalues[i, j] = boreholecolor



        baseimg = baseimg.transpose(Image.FLIP_TOP_BOTTOM)
        self.division = np.flip(self.division,1)


        arr = np.array(baseimg).astype(np.uint8)
        qimg = QImage(arr, baseimg.size[0],baseimg.size[1],QImage.Format_RGB888)

        self.pixmap = QPixmap.fromImage(qimg)


        self.addPixmap(self.pixmap)

        self.coordinate = None
        self.line = None

        self.itemset = []

        self.view = QGraphicsView(self)
        self.view.setMouseTracking(True)

    def getView(self):
        return self.view

    def mouseMoveEvent(self, event):
        super(AEMOverview, self).mouseMoveEvent(event)
        self.coordinate = event.scenePos()

        x = int(event.scenePos().x())
        y = int(event.scenePos().y())



        if x>=0 \
            and x<self.division.shape[0] \
            and y>=0 \
            and y<self.division.shape[1] \
            and self.line != self.division[x, y]:
            self.line = self.division[x, y]

            for i in self.itemset:
                self.removeItem(i)

            del self.itemset[:]

            for p in self.trace:
                if int(p[2])==self.line:
                    item = self.addRect(int(p[0]) - self.tracewidth*2, self.height - int(p[1]) - self.tracewidth*2,
                                              self.tracewidth*2, self.tracewidth*2, QPen(Qt.white), QBrush(Qt.white))
                    self.itemset.append(item)

        self.mouseMoved.emit()

    def getCoordinate(self):
        return self.coordinate.x(), self.coordinate.y()

    def getLine(self):
        return self.line