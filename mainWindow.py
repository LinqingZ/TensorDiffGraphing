import sys
import torch
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QFileDialog, QLineEdit, QPushButton, QFileDialog, QMessageBox
from heatmapWindow import Heatmap2DimenWindow, HeatmapMultiDimenWindow
from histogramWindow import Histogram2DimenWindow,  HistogramMultiDimenWindow

class MainWindow(QMainWindow):
    def __init__(self, file1_path=None, file2_path=None):
        super().__init__()
        self.setWindowTitle("File Value Extractor")
        self.tensor1= None
        self.tensor2= None
        
        self.file1_label = QLabel("File 1:")
        self.file1_lineedit = QLineEdit()
        self.file1_button = QPushButton("Browse")
        
        self.file2_label = QLabel("File 2:")
        self.file2_lineedit = QLineEdit()
        self.file2_button = QPushButton("Browse")
        
        self.heatmap_button = QPushButton("Graph Heatmap")
        self.histogram_button = QPushButton("Graph Histogram")
        
        self.layout = QVBoxLayout()
        
        file1_layout = QHBoxLayout()
        file1_layout.addWidget(self.file1_label)
        file1_layout.addWidget(self.file1_lineedit)
        file1_layout.addWidget(self.file1_button)
        
        file2_layout = QHBoxLayout()
        file2_layout.addWidget(self.file2_label)
        file2_layout.addWidget(self.file2_lineedit)
        file2_layout.addWidget(self.file2_button)
        
        self.layout.addLayout(file1_layout)
        self.layout.addLayout(file2_layout)
        self.layout.addWidget(self.heatmap_button)
        self.layout.addWidget(self.histogram_button)
        
        central_widget = QWidget()
        central_widget.setLayout(self.layout)
        self.setCentralWidget(central_widget)
        
        self.heatmap_window = None
        self.histogram_window = None
        
        self.file1_button.clicked.connect(self.browse_file1)
        self.file2_button.clicked.connect(self.browse_file2)
        self.heatmap_button.clicked.connect(self.open_heatmap_window)
        self.histogram_button.clicked.connect(self.open_histogram_window)
        
        if file1_path:
            self.file1_lineedit.setText(file1_path)
        if file2_path:
            self.file2_lineedit.setText(file2_path)
    
    def open_heatmap_window(self):
        if self.heatmap_window is not None:
                self.heatmap_window.close()
        if self.extract_values():
            if self.tensor1.ndim == 2:
                print("2d heatmap graph")
                self.heatmap_window = Heatmap2DimenWindow(self.tensor1, self.tensor2)
                self.heatmap_window.show()
            else:
                self.heatmap_window = HeatmapMultiDimenWindow(self.tensor1, self.tensor2)
                self.heatmap_window.show()
    
    def open_histogram_window(self):
        if self.histogram_window is not None:
                self.histogram_window.close()
        if self.extract_values():
            if self.tensor1.ndim == 2:
                print("2d histogram graph")
                self.histogram_window = Histogram2DimenWindow(self.tensor1, self.tensor2)
                self.histogram_window.show()
            else:
                self.histogram_window = HistogramMultiDimenWindow(self.tensor1, self.tensor2)
                self.histogram_window.show()
    
    def browse_file1(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File 1")
        if file_path:
            self.file1_lineedit.setText(file_path)
    
    def browse_file2(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File 2")
        if file_path:
            self.file2_lineedit.setText(file_path)
    
    def extract_values(self):
        file1_path = self.file1_lineedit.text()
        file2_path = self.file2_lineedit.text()
        if not file1_path.strip() and not file2_path.strip():
            QMessageBox.critical(self, "Error", "Please upload two valid files.")
            return
        if file1_path == file2_path:
            QMessageBox.critical(self, "Error", "Two files cannot be the same.")
            return
        
        try:
            data1 = np.load(file1_path)
            data2 = np.load(file2_path)
            
            if data1.shape != data2.shape:
                QMessageBox.critical(self, "Error", "Tensor sizes are not the same.")
                return
            
            print("data1.shape", data1.shape)
            self.tensor1 = torch.from_numpy(data1)
            self.tensor2 = torch.from_numpy(data2)
            return True
        except FileNotFoundError:
            QMessageBox.critical(self, "Error", "File not found.")
            return
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
