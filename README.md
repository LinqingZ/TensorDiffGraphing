# Project Name - Tensor Diff Visualization

## Description

This is a python visulization tool built by PyQt6. You can upload two tensor .npy files to check the different of two tensor. This app will be able to handle 2D tensor and multiple dimensions tensor on graphing of heatmap and histogram.

## Installation

To get started with the project, follow these steps:

1. Clone the repository to your local machine.
2. Navigate to the project directory.

Next, install the required dependencies by using pip:
```bash
pip install -r requirements.txt
```
Then run the mainWindow.py file, it will pop up a python app to upload two tensor files.

## Debug
Error Message
```
File "mainWindow.py", line 4, in <module>
    from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QFileDialog, QLineEdit, QPushButton, QFileDialog, QMessageBox
ImportError: libxkbcommon.so.0: cannot open shared object file: No such file or directory
```


```
File "mainWindow.py", line 4, in <module>
    from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QFileDialog, QLineEdit, QPushButton, QFileDialog, QMessageBox
ImportError: libEGL.so.1: cannot open shared object file: No such file or directory
```
Solution
```sudo apt-get install libgl1 libgle3-dev```
