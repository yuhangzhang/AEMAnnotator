import numpy as np

from skimage.draw import polygon

import matplotlib.pyplot as plt

from PyQt5.QtWidgets import QGraphicsScene

from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QPolygon
from PyQt5.QtGui import QPolygonF
from PyQt5.QtGui import QBrush
from PyQt5.QtGui import QPainterPath
from PyQt5.QtGui import QPen
from PyQt5.QtGui import QColor

from PyQt5.QtCore import Qt

from aemsectiondata import AEMSectionData
from dialogdropdown import DialogDropDown
from metric_learn import LMNN
from PIL import Image

# from geolmnn import GeoLMNN

class AEMSectionView(QGraphicsScene):
    def __init__(self, line):
        super(AEMSectionView, self).__init__()
        self.draw_switch = False
        self.dialog = DialogDropDown()
        self.pixmapunderground = QPixmap()
        self.pixmaptopdown = QPixmap()
        self.pixmapprediction = QPixmap()
        self.pixmapundergroundhandle = None
        self.pixmappredictionhandle = None
        self.line = line
        self.loaddatabase(1000,400)
        self.visiblelayer = {}

    def loaddatabase(self, width, height):
        self.geodata = AEMSectionData(self.line)
        arr = self.geodata.getimagetopdown(width, int(height / 4))
        qimg = QImage(arr, arr.shape[1], arr.shape[0], int(arr.nbytes/arr.shape[0]), QImage.Format_RGB888)#.rgbSwapped()
        print(qimg.size())
        self.pixmaptopdown = QPixmap(qimg)
        self.pixmaptopdownhandle = self.addPixmap(self.pixmaptopdown)
        self.pixmaptopdownhandle.moveBy(0, -self.pixmaptopdown.height()-20)

        arr_conductivity = self.geodata.getimageunderground('conductivity')
        arr_wii = self.geodata.getimageunderground('wii')
        arr_gravity = self.geodata.getimageunderground('gravity')
        self.arr = np.stack([arr_conductivity, arr_wii, arr_gravity], axis=2).astype(np.uint8)
        self.arr.fill(0)
        self.arrborehole = self.geodata.getimageunderground('borehole')
        qimg = QImage(self.arr, self.arr.shape[1], self.arr.shape[0], int(self.arr.nbytes/self.arr.shape[0]), QImage.Format_RGB888)#.rgbSwapped()

        self.pixmapunderground = QPixmap(qimg)
        self.pixmapundergroundhandle = self.addPixmap(self.pixmapunderground)

        #self.pixmapborehole = QPixmap(qimg)
        self.pixmapboreholehandle = None

    def fliplayer(self, layername):
        colorcode = [i for i, x in enumerate(self.getlayernames()) if x == layername][0]
        print(colorcode,'colorcode')

        if layername == 'borehole':
            if self.pixmapboreholehandle is None:
                qimg = QImage(self.arrborehole, self.arrborehole.shape[1], self.arrborehole.shape[0], int(self.arrborehole.nbytes / self.arrborehole.shape[0]), QImage.Format_RGBA8888)
                self.pixmapboreholehandle = self.addPixmap(QPixmap(qimg))
            else:
                self.removeItem(self.pixmapboreholehandle)
                self.pixmapboreholehandle = None
        else:
            if layername in self.visiblelayer:
                self.arr[:,:,colorcode%3] = 0
                del self.visiblelayer[layername]
            else:
                print(layername,'layername')
                layer = self.geodata.getimageunderground(layername)
                self.arr[:,:,colorcode%3] = layer
                self.visiblelayer[layername] = 1

            qimg = QImage(self.arr, self.arr.shape[1], self.arr.shape[0], int(self.arr.nbytes / self.arr.shape[0]), QImage.Format_RGB888)
            self.removeItem(self.pixmapundergroundhandle)
            self.pixmapundergroundhandle= self.addPixmap(QPixmap(qimg))


    def mousePressEvent(self, event):
        super(AEMSectionView, self).mousePressEvent(event)

        # prepare for a new crop
        if event.button() == Qt.LeftButton:
            self.draw_switch = True

            #print(event.scenePos().x(), event.scenePos().y())

            if event.scenePos().x()>=0 \
                    and event.scenePos().x()<self.pixmapunderground.width() \
                    and event.scenePos().y()>=0 \
                    and event.scenePos().y()<self.pixmapunderground.height():
                self.lastpos = event.pos()
                self.poly = [self.lastpos]
            else:
                self.lastpos = None
                self.poly = []

            self.pathset = []

    def mouseMoveEvent(self, event):
        super(AEMSectionView, self).mousePressEvent(event)

        pos = event.scenePos()

        if self.draw_switch == True \
                and pos.x()>=0 \
                and pos.x()<self.pixmapunderground.width() \
                and pos.y()>=0 \
                and pos.y()<self.pixmapunderground.height():
            if self.lastpos is not None:
                # show trace on the screen
                path = QPainterPath()
                path.setFillRule(Qt.WindingFill)
                path.moveTo(self.lastpos)
                path.lineTo(pos)
                self.pathset.append(self.addPath(path, pen=QPen(Qt.white)))
                self.poly.append(pos)  # keep vertex for label generation later

            self.lastpos = pos  # update

    def mouseReleaseEvent(self, event):
        super(AEMSectionView, self).mousePressEvent(event)

        pos = event.scenePos()

        if event.button() == Qt.LeftButton:
            self.draw_switch = False

            if pos.x()>=0 \
            and pos.x()<self.pixmapunderground.width() \
            and pos.y()>=0 \
            and pos.y()<self.pixmapunderground.height():
                label = self.dialog.gettext()   #ask for label

                if label[1]==True and len(label[0])>0 and len(self.poly)>0:  #if user input a label


                    # save the label on backend
                    if len(self.poly)>1:
                        # point the label on screen
                        poly = QPolygon()
                        for p in self.poly:
                            poly.append(p.toPoint())
                            # print(p)
                        brush = QBrush()
                        labelcolor = QColor(*[c*255 for c in plt.get_cmap('tab10').colors[int(label[0])-1]])
                        brush.setColor(labelcolor)
                        brush.setStyle(Qt.SolidPattern)
                        self.addPolygon(QPolygonF(poly), pen=QPen(labelcolor), brush=brush)
                        x, y = polygon([p.toPoint().x() for p in self.poly],[p.toPoint().y() for p in self.poly])
                    else:
                        self.addEllipse(self.poly[0].x(),self.poly[0].y(),2,2,pen=QPen(Qt.red))
                        x = self.poly[0].toPoint().x()
                        y = self.poly[0].toPoint().y()
                    self.geodata.manuallabel[y, x] = int(label[0])

            # remove the trace painted so far
            for p in self.pathset:
                self.removeItem(p)

    def export(self, filename):
        fw = open(filename, 'w')
        points, labels = self.geodata.get_annotated_point()
        for i in range(len(points)):
            np.savetxt(fw, np.append(points[i][0:-2],labels[i]).reshape(1, -1), fmt='%s')
        fw.close()



    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            if self.pixmapundergroundhandle is not None:
                self.pixmapundergroundhandle.setZValue(1)


    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space:
            if self.pixmapundergroundhandle is not None:
                self.pixmapundergroundhandle.setZValue(-1)

    def showprediction(self):
        arr = self.geodata.get_prediction(GeoLMNN(3))
        arr[:,:,3] = 255
        qimg = QImage(arr.astype(np.uint8), arr.shape[1], arr.shape[0], QImage.Format_RGBA8888)  # .rgbSwapped()
        self.pixmapprediction = QPixmap(qimg)
        self.pixmappredictionhandle = self.addPixmap(self.pixmapprediction)

        return

    def getlayernames(self):
        return self.geodata.getlayernames()