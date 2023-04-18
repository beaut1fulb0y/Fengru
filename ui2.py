import sys, os, subprocess,matplotlib
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QScrollArea, QFileDialog, QPushButton, QLineEdit, QMessageBox
from PyQt5.QtGui import QPixmap
import os
import PIL.Image as Image
import torch
from utils import visualize_heatmap
import matplotlib.pyplot as plt


class ImageWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.title = '下胫腓联合不稳症状诊断'
        self.left = 200
        self.top = 200
        self.width = 1200
        self.height = 800
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.scroll1 = QScrollArea(self)
        self.scroll1.setWidgetResizable(True)
        self.scrollContent1 = QWidget(self.scroll1)
        self.scrollLayout1 = QVBoxLayout(self.scrollContent1)
        self.scroll1.setWidget(self.scrollContent1)

        self.scroll2 = QScrollArea(self)
        self.scroll2.setWidgetResizable(True)
        self.scrollContent2 = QWidget(self.scroll2)
        self.scrollLayout2 = QVBoxLayout(self.scrollContent2)
        self.scroll2.setWidget(self.scrollContent2)

        self.scroll3 = QScrollArea(self)
        self.scroll3.setWidgetResizable(True)
        self.scrollContent3 = QWidget(self.scroll3)
        self.scrollLayout3 = QVBoxLayout(self.scrollContent3)
        self.scroll3.setWidget(self.scrollContent3)

        self.clearButton = QPushButton('清空', self)
        self.clearButton.clicked.connect(self.clearImages)

        self.openButton1 = QPushButton('打开1', self)
        self.openButton1.clicked.connect(self.openFile1)

        self.openButton2 = QPushButton('打开2', self)
        self.openButton2.clicked.connect(self.openFile2)

        self.openButton3 = QPushButton('打开3', self)
        self.openButton3.clicked.connect(self.openFile3)

        self.saveButton = QPushButton('另存', self)
        self.saveButton.clicked.connect(self.saveImages)

        self.saveLineEdit = QLineEdit(self)
        self.saveLineEdit.setReadOnly(True)

        self.saveFolderButton = QPushButton('选择路径', self)
        self.saveFolderButton.clicked.connect(self.selectFolder)

        self.patientLineEdit = QLineEdit(self)
        self.patientLineEdit.setPlaceholderText('请输入患者id')

        self.runButton = QPushButton('运行', self)
        self.runButton.clicked.connect(self.predict)



        self.buttonLayout = QHBoxLayout()
        self.buttonLayout.addWidget(self.clearButton)
        self.buttonLayout.addWidget(self.openButton1)
        self.buttonLayout.addWidget(self.openButton2)
        self.buttonLayout.addWidget(self.openButton3)
        self.buttonLayout.addWidget(self.saveButton)
        self.buttonLayout.addWidget(self.saveLineEdit)
        self.buttonLayout.addWidget(self.saveFolderButton)
        self.buttonLayout.addWidget(self.patientLineEdit)
        self.buttonLayout.addWidget(self.runButton)


        self.imageLayout = QHBoxLayout()
        self.imageLayout.addWidget(self.scroll1)
        self.imageLayout.addWidget(self.scroll2)
        self.imageLayout.addWidget(self.scroll3)

        self.layout = QVBoxLayout(self)
        self.layout.addLayout(self.imageLayout)
        self.layout.addLayout(self.buttonLayout)
        self.show()

        self.resultLabel = QLabel('诊断结果：', self)

        self.labelLayout = QHBoxLayout()
        self.labelLayout.addWidget(self.resultLabel)

        self.layout.addLayout(self.labelLayout)
        self.image_paths1 = []
        self.image_paths2 = []
        self.image_paths3 = []

    def openFile1(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self,"Open Files", "", "Images (*.png *.xpm *.jpg *.jpeg *.bmp *.gif);;Python Files (*.py)", options=options)
        if files:
            for file in files:
                if file.endswith('.py'):
                    self.runPython(file)
                else:
                    pixmap = QPixmap(file)
                    label = QLabel(self.scrollContent1)
                    label.setPixmap(pixmap)
                    self.scrollLayout1.addWidget(label)
                    self.image_paths1.append(file)

    def openFile2(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self,"Open Files", "", "Images (*.png *.xpm *.jpg *.jpeg *.bmp *.gif);;Python Files (*.py)", options=options)
        if files:
            for file in files:
                if file.endswith('.py'):
                    self.runPython(file)
                else:
                    pixmap = QPixmap(file)
                    label = QLabel(self.scrollContent2)
                    label.setPixmap(pixmap)
                    self.scrollLayout2.addWidget(label)
                    self.image_paths2.append(file)

    def openFile3(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self, "Open Files", "","Images (*.png *.xpm *.jpg *.jpeg *.bmp *.gif);;Python Files (*.py)", options=options)
        if files:
            for file in files:
                if file.endswith('.py'):
                    self.runPython(file)
                else:
                    pixmap = QPixmap(file)
                    label = QLabel(self.scrollContent3)
                    label.setPixmap(pixmap)
                    self.scrollLayout3.addWidget(label)
                    self.image_paths3.append(file)
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
        count1 = self.scrollLayout1.count()
        count2 = self.scrollLayout2.count()
        count3 = self.scrollLayout3.count()
        if count1 == 0 and count2 == 0 and count3 == 0:
            QMessageBox.warning(self, 'Warning', 'Please select images in at least one area!')
            return
        for i in range(count1):
            item = self.scrollLayout1.itemAt(i).widget()
            pixmap = item.pixmap()
            if pixmap:
                file_name = os.path.join(folder_path, '{}-image{}-1.jpg'.format(patient_id, i))
                pixmap.save(file_name, 'jpg')
        for i in range(count2):
            item = self.scrollLayout2.itemAt(i).widget()
            pixmap = item.pixmap()
            if pixmap:
                file_name = os.path.join(folder_path, '{}-image{}-2.jpg'.format(patient_id, i))
                pixmap.save(file_name, 'jpg')
        for i in range(count3):
            item = self.scrollLayout3.itemAt(i).widget()
            pixmap = item.pixmap()
            if pixmap:
                file_name = os.path.join(folder_path, '{}-image{}-3.jpg'.format(patient_id, i))
                pixmap.save(file_name, 'jpg')

    # def runProgram(self):
    #     folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder')
    #     if folder_path:
    #         for file in os.listdir(folder_path):
    #             if file.endswith('.py'):
    #                 self.runPython(os.path.join(folder_path, file))
    #                 break
    #
    # def runPython(self, file):
    #     subprocess.Popen(['python', file])

    def predict(self):
        # pathi stands for path to image of viewi
        path1 = self.image_paths1[0]  # 获取读取时图片的路径
        path2 = self.image_paths2[0]
        path3 = self.image_paths3[0]

        image1 = Image.open(path1)
        image2 = Image.open(path2)
        image3 = Image.open(path3)

        png_path1 = os.path.join(os.path.dirname(__file__), 'runs', '1.png')
        png_path2 = os.path.join(os.path.dirname(__file__), 'runs', '2.png')
        png_path3 = os.path.join(os.path.dirname(__file__), 'runs', '3.png')

        predict1, heatmap1 = visualize_heatmap(1, image1)
        predict2, heatmap2 = visualize_heatmap(2, image2)
        predict3, heatmap3 = visualize_heatmap(3, image3)

        # 1 for afflicted 0 for unafflicted
        status = torch.argmax(predict1 + predict2 + predict3).item()
        self.resultLabel.setText('诊断结果：{}'.format(status))

        # 显示热力图
        plt.imshow(heatmap1)
        plt.show()
        plt.imshow(heatmap2)
        plt.show()
        plt.imshow(heatmap3)
        plt.show()
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageWindow()
    sys.exit(app.exec_())
