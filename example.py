import sys


from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QGraphicsView

from aemoverdata import AEMOverData
from aemoverview import AEMOverview



class Menu(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Overview")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)



        self.data = AEMOverData()
        baseimg, trace, borehole = self.data.getOverview()
        self.overview = AEMOverview(baseimg, trace, borehole)
        layout.addWidget(self.overview.getView())

        self.sb = self.statusBar()
        self.sb.showMessage(self.overview.coordinate)
        self.overview.mouseMoved.connect(self.refreshcoord)


    def refreshcoord(self):
        self.sb.showMessage('Line '+str(self.overview.getLine()))


if __name__ == '__main__':


    app = QApplication(sys.argv)
    widget = Menu()
    widget.resize(1024, 768)
    widget.show()
    sys.exit(app.exec_())