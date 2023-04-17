import sys, os, subprocess
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QScrollArea, QFileDialog, QPushButton, QLineEdit, QMessageBox
from PyQt5.QtGui import QPixmap

class ImageWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.title = '下胫腓联合不稳症状诊断'
        self.left = 200
        self.top = 200
        self.width = 3600
        self.height = 1200
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scrollContent = QWidget(self.scroll)
        self.scrollLayout = QVBoxLayout(self.scrollContent)
        self.scroll.setWidget(self.scrollContent)

        self.clearButton = QPushButton('清空', self)
        self.clearButton.clicked.connect(self.clearImages)

        self.openButton = QPushButton('打开', self)
        self.openButton.clicked.connect(self.openFile)

        self.saveButton = QPushButton('另存', self)
        self.saveButton.clicked.connect(self.saveImages)

        self.saveLineEdit = QLineEdit(self)
        self.saveLineEdit.setReadOnly(True)

        self.saveFolderButton = QPushButton('选择路径', self)
        self.saveFolderButton.clicked.connect(self.selectFolder)

        self.patientLineEdit = QLineEdit(self)
        self.patientLineEdit.setPlaceholderText('请输入患者id')

        self.runButton = QPushButton('运行', self)
        self.runButton.clicked.connect(self.runProgram)

        self.buttonLayout = QHBoxLayout()
        self.buttonLayout.addWidget(self.clearButton)
        self.buttonLayout.addWidget(self.openButton)
        self.buttonLayout.addWidget(self.saveButton)
        self.buttonLayout.addWidget(self.saveLineEdit)
        self.buttonLayout.addWidget(self.saveFolderButton)
        self.buttonLayout.addWidget(self.patientLineEdit)
        self.buttonLayout.addWidget(self.runButton)

        self.imageLayout = QHBoxLayout()
        self.imageLayout.addStretch()
        self.imageLayout.addWidget(self.scroll)
        self.imageLayout.addStretch()

        self.layout = QVBoxLayout(self)
        self.layout.addLayout(self.imageLayout)
        self.layout.addLayout(self.buttonLayout)
        self.show()

        self.resultLabel = QLabel('诊断结果：', self)
        self.accuracyLabel = QLabel('准确率：', self)

        self.labelLayout = QHBoxLayout()
        self.labelLayout.addWidget(self.resultLabel)
        self.labelLayout.addWidget(self.accuracyLabel)

        self.layout.addLayout(self.labelLayout)

    def openFile(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self,"Open Files", "", "Images (*.png *.xpm *.jpg *.jpeg *.bmp *.gif);;Python Files (*.py)", options=options)
        if files:
            for file in files:
                if file.endswith('.py'):
                    self.runPython(file)
                else:
                    pixmap = QPixmap(file)
                    label = QLabel(self.scrollContent)
                    label.setPixmap(pixmap)
                    self.scrollLayout.addWidget(label)

    def clearImages(self):
        for i in reversed(range(self.scrollLayout.count())):
            self.scrollLayout.itemAt(i).widget().setParent(None)

    def selectFolder(self):
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if folder_path:
            self.saveLineEdit.setText(folder_path)

    def saveImages(self):
        folder_path = self.saveLineEdit.text()
        if not os.path.isdir(folder_path):
            QMessageBox.warning(self, 'Warning', 'The folder path is not valid!')
            return
        patient_id = self.patientLineEdit.text()
        count = self.scrollLayout.count()
        for i in range(count):
            item = self.scrollLayout.itemAt(i).widget()
            pixmap = item.pixmap()
            if pixmap:
                file_name = os.path.join(folder_path, '{}-image{}.jpg'.format(patient_id, i))
                pixmap.save(file_name, 'jpg')

    def runProgram(self):
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if folder_path:
            for file in os.listdir(folder_path):
                if file.endswith('.py'):
                    self.runPython(os.path.join(folder_path, file))
                    break

    def runPython(self, file):
        subprocess.Popen(['python', file])

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageWindow()
    sys.exit(app.exec_())
