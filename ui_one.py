import sys, subprocess
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QScrollArea, QFileDialog, QPushButton, QLineEdit, QMessageBox
from PyQt5.QtGui import QPixmap
import os
import PIL.Image as Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from model import CustomResNet18
from utils import visualize_heatmap
import matplotlib.pyplot as plt

class ImageWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.image_path=[]
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
        self.runButton.clicked.connect(self.predict)

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

        self.labelLayout = QHBoxLayout()
        self.labelLayout.addWidget(self.resultLabel)

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
                    self.image_path.append(file)

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
        count1 = self.scrollLayout.count()

        if count1 == 0 :
            QMessageBox.warning(self, 'Warning', 'Please select images in at least one area!')
            return
        for i in range(count1):
            item = self.scrollLayout.itemAt(i).widget()
            if isinstance(item, QLabel):
                pixmap = item.pixmap()
                if pixmap:
                    file_name = os.path.join(folder_path, '{}-image{}.jpg'.format(patient_id, i))
                    pixmap.save(file_name, 'jpg')


    def runPython(self, file):
        subprocess.Popen(['python', file])

    def predict(self):
        model_sep = CustomResNet18(num_classes=3)
        state_dict = torch.load('parameters/best.pth', map_location='cpu')
        model_sep.load_state_dict(state_dict)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.14770294, 0.14736584, 0.14737843), (0.14732725, 0.14687528, 0.14688413)),
        ])

        pred_list = []
        heatmap_list = []

        for path in self.image_path:
            image = Image.open(path)
            image_input = transform(image).unsqueeze(0)
            out = F.softmax(model_sep(image_input)).squeeze(dim=0)
            view = torch.argmax(out).item() + 1

            pred, heatmap = visualize_heatmap(view, image)
            pred_list.append(pred)
            heatmap_list.append(heatmap)

        # 1 for afflicted 0 for unafflicted
        sum_pred = torch.tensor([0, 0], dtype=torch.float32)
        for i in pred_list:
            sum_pred += i

        ave_pred = sum_pred / len(pred_list)
        status = torch.argmax(ave_pred).item()
        self.resultLabel.setText('诊断结果：{}'.format(status))


        for heatmap in heatmap_list:
            plt.imshow(heatmap)
            plt.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageWindow()
    sys.exit(app.exec_())
