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
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QVBoxLayout

from PIL import Image
from PIL.ImageDraw import  Draw
from PIL.ImageQt import ImageQt

from aemsectionview import AEMSectionView

class AEMOverview(QGraphicsScene):
    mouseMoved = pyqtSignal()

    def __init__(self, baseimg, trace, borehole,
                 tracewidth=1, effectivewidth=8, boreholesize=4, tracecolor=(255,255,0), boreholecolor=(255,0,0)):
        super(AEMOverview, self).__init__()

        self.division = np.zeros(baseimg.size, dtype=np.int)
        self.division.fill(-1)

        self.height = baseimg.size[1]
        self.tracewidth = tracewidth
        self.trace = {'-1':[]}
        self.coordinate = None
        self.line = -1

        self.itemset = []

        pixelvalues = baseimg.load()

        draw = Draw(baseimg)

        for point in trace:
            x = int(point[0])
            y = int(point[1])
            id= point[2]
            if id in self.trace:
                self.trace[id].append([x,y])
            else:
                self.trace[id] = [[x,y]]

            draw.ellipse((x-tracewidth, y-tracewidth, x+tracewidth, y+tracewidth), fill=tracecolor)
            self.division[x-tracewidth*effectivewidth : x+tracewidth*effectivewidth, y-tracewidth*effectivewidth : y+tracewidth*effectivewidth] = id




        for point in borehole:
            x = point[0]
            y = point[1]
            if (x >= 0 and x < baseimg.size[0] and y >= 0 and y < baseimg.size[1]):
                draw.ellipse((x - boreholesize, y - boreholesize, x + boreholesize, y + boreholesize), fill=boreholecolor)

        baseimg = baseimg.transpose(Image.FLIP_TOP_BOTTOM)
        self.division = np.flip(self.division,1)

        arr = np.array(baseimg).astype(np.uint8)
        qimg = QImage(arr, baseimg.size[0],baseimg.size[1],QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(qimg)
        self.addPixmap(self.pixmap)



        self.view = QGraphicsView(self)
        self.view.setMouseTracking(True)

    def getView(self):
        return self.view

    def mouseDoubleClickEvent(self, event):
        if self.line<0:
            pass
        else:
            self.annotator = AEMSectionView(self.line)
            self.window = QMainWindow()
            self.window.setWindowTitle("Line"+str(self.line))

            central_widget = QWidget()
            self.window.setCentralWidget(central_widget)
            layout = QVBoxLayout(central_widget)

            self.view = QGraphicsView(self.annotator)

            layout.addWidget(self.view)

            self.window.show()

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

            for p in self.trace[str(self.line)]:
                item = self.addEllipse(int(p[0])-self.tracewidth, self.height-int(p[1])-self.tracewidth-1, self.tracewidth*2, self.tracewidth*2, QPen(Qt.white), QBrush(Qt.white))
                self.itemset.append(item)

        self.mouseMoved.emit()

    def getCoordinate(self):
        return self.coordinate.x(), self.coordinate.y()

    def getLine(self):
        if self.line<0:
            return ''
        else:
            return 'line '+str(self.line)