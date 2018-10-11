import sys

from PyQt5.QtWidgets import QWidget

from PyQt5.QtWidgets import QGraphicsView
from PyQt5.QtWidgets import QGraphicsScene
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QAbstractItemView
from PyQt5.QtWidgets import QListView
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItemModel
from PyQt5.QtGui import QStandardItem
from PyQt5.QtWidgets import QMainWindow
from aemsectionview import AEMSectionView


class AEMSectionWindow(QMainWindow):
    def __init__(self, lineNumber):
        super(AEMSectionWindow, self).__init__()

        self.setWindowTitle("Line" + str(lineNumber))

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        self.view = AEMSectionView(lineNumber)

        layout.addWidget(QGraphicsView(self.view))


        self.wavelist = QListView()
        self.wavelistmodel = QStandardItemModel()

        for name in self.view.getlayernames():
            item = QStandardItem(name)
            item.setSelectable(True)
            self.wavelistmodel.appendRow(item)

        self.wavelist.setModel(self.wavelistmodel)
        self.wavelist.setSelectionMode(QAbstractItemView.MultiSelection)
        self.wavelist.clicked.connect(self.listclick)

        layout.addWidget(self.wavelist)

        self.show()

    def listclick(self, index):
        layername = self.wavelistmodel.itemFromIndex(index).text()
        self.view.fliplayer(layername)