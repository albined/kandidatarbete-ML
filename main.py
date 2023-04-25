import os
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QCheckBox,  QLabel, QLineEdit, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QGridLayout, QSpacerItem, QSizePolicy, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore
import extra_filer.extra_tools as tools

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.svd_image_path = None

        self.setWindowTitle("My Project")
        self.setStyleSheet("background-color: #333333; color: white;")

        self.nn_label = QLabel("Neural network image classification")
        self.nn_label.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 10px;")

        self.nn_input1_label = QLabel("Input 1:")
        self.nn_input1 = QLineEdit()
        self.nn_input2_label = QLabel("Input 2:")
        self.nn_input2 = QLineEdit()

        self.nn_button_train = QPushButton("Train")
        self.nn_button_train.clicked.connect(self.train_nn)

        self.nn_button_test = QPushButton("Test")
        self.nn_button_test.clicked.connect(self.test_nn)

        self.nn_checkboxes = [QCheckBox("Linear"), QCheckBox("CNN"), QCheckBox("Score"), QCheckBox("Uk"), QCheckBox("Vk")]
        for i, checkbox in enumerate(self.nn_checkboxes):
            checkbox.clicked.connect(lambda state, n=i: self.checkbox_clicked(n))

        self.nn_description = QLabel("Fill the input fields with comma separated values: x1, x2, ...")
        self.nn_description.setStyleSheet("font-size: 12px;")

        self.nn_layout = QVBoxLayout()
        self.nn_layout.addWidget(self.nn_label)
        for checkbox in self.nn_checkboxes:
            self.nn_layout.addWidget(checkbox)
        self.nn_layout.addWidget(self.nn_description)
        self.nn_layout.addWidget(self.nn_input1_label)
        self.nn_layout.addWidget(self.nn_input1)
        self.nn_layout.addWidget(self.nn_input2_label)
        self.nn_layout.addWidget(self.nn_input2)
        self.nn_layout.addWidget(self.nn_button_train)
        self.nn_layout.addWidget(self.nn_button_test)


        self.pca_label = QLabel("PCA image classification")
        self.pca_label.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 10px;")

        self.pca_input1_label = QLabel("Input 1:")
        self.pca_input1 = QLineEdit()
        self.pca_input2_label = QLabel("Input 2:")
        self.pca_input2 = QLineEdit()
        self.pca_button = QPushButton("Run")
        self.pca_button.clicked.connect(self.run_pca)

        self.pca_layout = QVBoxLayout()
        self.pca_layout.addWidget(self.pca_label)
        self.pca_layout.addWidget(self.pca_input1_label)
        self.pca_layout.addWidget(self.pca_input1)
        self.pca_layout.addWidget(self.pca_input2_label)
        self.pca_layout.addWidget(self.pca_input2)
        self.pca_layout.addWidget(self.pca_button)

        self.svd_label = QLabel("SVD image compression")
        self.svd_label.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 10px;")
        self.svd_description = QLabel("Enter the number of singular values to keep (k)")
        self.svd_description.setStyleSheet("font-size: 12px;")

        self.svd_input_label = QLabel("Input:")
        self.svd_input = QLineEdit()
        # self.svd_input.setReadOnly(True)
        self.svd_import_button = QPushButton("Import")
        self.svd_import_button.clicked.connect(self.import_image)
        self.svd_run_button = QPushButton("Run")
        self.svd_run_button.clicked.connect(self.run_svd)
        self.svd_image_label = QLabel()
        self.svd_image_label.setFixedSize(200, 200)
        self.svd_image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.svd_image_label.setStyleSheet("border: 1px solid white;")

        self.svd_image_output_label = QLabel()
        self.svd_image_output_label.setFixedSize(200, 200)
        self.svd_image_output_label.setAlignment(QtCore.Qt.AlignCenter)
        self.svd_image_output_label.setStyleSheet("border: 1px solid white;")

        self.svd_layout = QGridLayout()
        self.svd_layout.addWidget(self.svd_label, 0, 0, 1, 4)
        self.svd_layout.addWidget(self.svd_description, 1, 0, 1, 4)
        self.svd_layout.addWidget(self.svd_input_label, 2, 0,)
        self.svd_layout.addWidget(self.svd_input, 2, 1, 1, 3)
        self.svd_layout.addWidget(self.svd_import_button, 3, 0)
        self.svd_layout.addWidget(self.svd_run_button, 3, 1, 1, 3)
        self.svd_layout.addWidget(self.svd_image_label, 4, 0, 1, 2)
        self.svd_layout.addWidget(self.svd_image_output_label, 4, 2)
        self.svd_layout.setColumnStretch(1, 1)

        self.layout = QVBoxLayout()
        self.layout.addLayout(self.nn_layout)
        self.layout.addLayout(self.pca_layout)
        self.layout.addLayout(self.svd_layout)

        self.setLayout(self.layout)
        self.import_image(os.path.join(tools.get_project_root(), 'svd_image_compression', 'zelda.png'))

        self.show()

    def train_nn(self):
        input1 = self.nn_input1.text()
        input2 = self.nn_input2.text()
        # call the function that handles the neural network classification

    def test_nn(self):
        input1 = self.nn_input1.text()
        input2 = self.nn_input2.text()
        # call the function that handles the neural network classification

    def run_pca(self):
        input1 = self.pca_input1.text()
        input2 = self.pca_input2.text()
        # call the function that handles the PCA image classification

    def import_image(self, file_name=None):
        if file_name is None:
            file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if file_name:
            # self.svd_input.setText(file_name)
            self.svd_image_path = file_name
            pixmap = QPixmap(file_name)
            self.svd_image_label.setPixmap(pixmap.scaled(self.svd_image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def run_svd(self):
        input_path = self.svd_input.text()
        # call the function that handles the SVD image compression

    def checkbox_clicked(self, n):
        for idx, checkbox in enumerate(self.nn_checkboxes):
            if idx != n:
                checkbox.setChecked(False)
            else:
                checkbox.setChecked(True)
        input1_dict = [
            "Layer sizes (x, x/2)",
            "Linear layer size (x)",
            "Linear layer sizes (x, x/2)",
            "Linear layer sizes (x, x/2)",
            "Linear layer sizes (x, x/2)"
        ]
        input2_dict = [
            "Leave this empty",
            "Convolutional layer sizes (x, x/2)",
            "k (x)",
            "k (x)",
            "k (x)"
        ]
        self.nn_input1_label.setText(input1_dict[n])
        self.nn_input2_label.setText(input2_dict[n])



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())