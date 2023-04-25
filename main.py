import os
import sys
import warnings

from PyQt5.QtGui import QIntValidator
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QWidget, QCheckBox, QLabel, QLineEdit, QVBoxLayout, QHBoxLayout, QPushButton, \
    QFileDialog, QGridLayout, QSpacerItem, QSizePolicy, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore
import extra_filer.extra_tools as tools
import svd_image_compression.image_compression as svd_imagecompression
import neural_network_image_classification.neural_network_testing as nn_testing
import threading


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.svd_image_path = None

        self.setWindowTitle("Kandidatarbete")
        self.setStyleSheet("background-color: #333333; color: white;")

        self.nn_label = QLabel("Neural network image classification")
        self.nn_label.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 10px;")

        self.nn_input1_label = QLabel("Input 1:")
        self.nn_input1 = QLineEdit()
        self.nn_input2_label = QLabel("Input 2:")
        self.nn_input2 = QLineEdit()

        self.nn_button_train = QPushButton("Train and Test")
        self.nn_button_train.clicked.connect(self.train_nn)

        self.nn_button_test = QPushButton("Only Test")
        self.nn_button_test.clicked.connect(self.test_nn)

        self.nn_checkboxes = [QCheckBox("Linear"), QCheckBox("CNN"), QCheckBox("Score"), QCheckBox("Uk"),
                              QCheckBox("Vk")]
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
        self.svd_input = MyLineEdit()
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
        self.svd_layout.addWidget(self.svd_input_label, 2, 0, )
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
        self.import_image(False, os.path.join(tools.get_project_root(), 'svd_image_compression', 'zelda.png'))

        self.show()

    def train_nn(self):
        self.test_train_nn(onlytest=False)

    def test_nn(self):
        print('Testing NN')
        self.test_train_nn(onlytest=True)

    def test_train_nn(self, onlytest):
        print('Training NN')
        input1 = self.nn_input1.text()
        input2 = self.nn_input2.text()

        checked = [checkbox.checkState() for checkbox in self.nn_checkboxes]

        train_test_func = None
        plot_func = None

        if checked[0]:
            input1 = self.check_input(input1, 1024)
            if input1 is None:
                return
            train_test_func = lambda: nn_testing.test_normal_matrix(onlytest=onlytest, hidden_layer_sizes=input1)
            plot_func = lambda: nn_testing.plot_normal_result()
        elif checked[1]:
            input1 = self.check_input(input1, 1024)
            input2 = self.check_input(input2, 128)
            if input1 is None or input2 is None:
                return
            train_test_func = lambda: nn_testing.test_cnn_matrix(onlytest=onlytest, linear_layer_sizes=input1,
                                                                 convolutional_layer_sizes=input2)
            plot_func = lambda: nn_testing.plot_cnn_result()
        elif checked[2]:
            input1 = self.check_input(input1, 1024)
            input2 = self.check_input(input2, 784)
            if input1 is None or input2 is None:
                return
            train_test_func = lambda: nn_testing.test_score_matrix(onlytest=onlytest, hidden_layer_sizes=input1,
                                                                   k_list=input2)
            plot_func = lambda: nn_testing.plot_result_matrix('score')
        elif checked[3]:
            input1 = self.check_input(input1, 1024)
            input2 = self.check_input(input2, 28)
            if input1 is None or input2 is None:
                return
            train_test_func = lambda: nn_testing.test_u_v_matrix('U', onlytest=onlytest, hidden_layer_sizes=input1,
                                                                 k_list=input2)
            plot_func = lambda: nn_testing.plot_result_matrix('U')
        elif checked[4]:
            input1 = self.check_input(input1, 1024)
            input2 = self.check_input(input2, 28)
            if input1 is None or input2 is None:
                return
            train_test_func = lambda: nn_testing.test_u_v_matrix('V', onlytest=onlytest, hidden_layer_sizes=input1,
                                                                 k_list=input2)
            plot_func = lambda: nn_testing.plot_result_matrix('V')

        # thread = threading.Thread(target=train_test_func)
        # thread.start()
        worker = NNWorker(train_test_func)
        worker.finished.connect(plot_func)
        worker.run()

    def run_pca(self):
        input1 = self.pca_input1.text()
        input2 = self.pca_input2.text()
        # call the function that handles the PCA image classification


    def import_image(self, state, file_name=None):
        if file_name is None:
            file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if file_name:
            # self.svd_input.setText(file_name)
            self.svd_image_path = file_name
            pixmap = QPixmap(file_name)
            self.svd_image_label.setPixmap(
                pixmap.scaled(self.svd_image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))


    def run_svd(self):
        input_path = self.svd_image_path
        k = self.svd_input.text()
        if k is None or len(k) == 0:
            warnings.warn('Input field can\'t be empty')
            return
        k = int(k)
        worker = SVDWorker(svd_imagecompression.compress_image, input_path, k)
        worker.finished.connect(self.update_result_image)
        worker.run()
        # call the function that handles the SVD image compression


    def update_result_image(self, path):
        pixmap = QPixmap(path)
        self.svd_image_output_label.setPixmap(
            pixmap.scaled(self.svd_image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))


    def checkbox_clicked(self, n):
        for idx, checkbox in enumerate(self.nn_checkboxes):
            if idx != n:
                checkbox.setChecked(False)
            else:
                checkbox.setChecked(True)
        input1_titles = [
            "Layer sizes (x, x/2)",
            "Linear layer size (x)",
            "Linear layer sizes (x, x/2)",
            "Linear layer sizes (x, x/2)",
            "Linear layer sizes (x, x/2)"
        ]
        input2_titles = [
            "Leave this empty",
            "Convolutional layer sizes (x, x/2)",
            "k (x)",
            "k (x)",
            "k (x)"
        ]
        input1_defaults = [
            "8, 16, 32, 64, 128, 256",
            "8, 16, 32, 64, 128, 256",
            "8, 16, 32, 64, 128, 256",
            "16, 64, 256",
            "16, 64, 256"
        ]
        input2_defaults = [
            "",
            "8, 16, 32",
            "1, 2, 3, 5, 8, 14, 20, 28, 64, 128",
            "1, 2, 3, 5, 8, 14, 20",
            "1, 2, 3, 5, 8, 14, 20"
        ]

        self.nn_input1_label.setText(input1_titles[n])
        self.nn_input2_label.setText(input2_titles[n])
        self.nn_input1.setText(input1_defaults[n])
        self.nn_input2.setText(input2_defaults[n])


    def check_input(self, input: str, upper_bound: int):
        # Split the input string into a list of strings
        input_list = input.split(',')

        # Check that all the elements in the list are integers
        try:
            input_list = [int(x) for x in input_list]
        except ValueError:
            print("Invalid input: not all elements are integers")
            return None

        # Check that all the integers are between 0 and upper_bound
        for x in input_list:
            if x <= 0 or x >= upper_bound:
                print("Invalid input: integer out of range")
                return None

        # Check that there is at least one element in the list
        if len(input_list) == 0:
            print("Invalid input: empty list")
            return None

        # Return the list of integers if everything checks out
        return input_list


class SVDWorker(QThread):
    finished = pyqtSignal(str)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        result = self.func(*self.args, **self.kwargs)
        self.finished.emit(result)

class NNWorker(QThread):
    finished = pyqtSignal()

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        self.func(*self.args, **self.kwargs)
        self.finished.emit()


class MyLineEdit(QLineEdit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setValidator(QIntValidator(1, 500))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
