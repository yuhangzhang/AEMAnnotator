
from PyQt5.QtWidgets import QDialog
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtWidgets import QComboBox
from PyQt5.QtGui import QIntValidator
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialogButtonBox

import re

class DialogDropDown(QDialog):

    def __init__(self):
        super(DialogDropDown, self).__init__()

        self.text = '1'

        layout = QVBoxLayout(self)

        self.dd = QComboBox()
        self.dd.addItem('1: Fresh bedrock Proterozoic')
        self.dd.addItem('2: Moderately weathered bedrock Proterozoic')
        self.dd.addItem('3: Very highly weathered bedrock Proterozoic')
        self.dd.addItem('4: Semi-consolidated sediments Cenozoic')
        self.dd.addItem('5: ')
        self.dd.addItem('6: ')
        self.dd.addItem('7: Semi-consolidated sediments Cenozoic')
        self.dd.addItem('8: Bedrock moderately resistive Palaeozoic')
        self.dd.addItem('9: Bedrock highly resistive Palaeozoic')
        self.dd.addItem('10: Very highly weathered bedrock Palaeozoic')
        self.dd.setCurrentIndex(0);

        self.dd.activated[str].connect(self.settext)

        layout.addWidget(self.dd)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def settext(self,txt):
        self.text = re.search('([0-9]+)(:.*)',txt).group(1)

    @staticmethod
    def gettext():
        dialog = DialogDropDown()
        result = dialog.exec_()
        return(dialog.text ,result==QDialog.Accepted)

