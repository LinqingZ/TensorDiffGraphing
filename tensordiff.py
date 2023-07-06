import sys
import numpy as np
import mplcursors
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QApplication, QMainWindow, QComboBox, QSizePolicy, QWidget, QCheckBox, QHBoxLayout, QVBoxLayout, QLabel, QFileDialog, QLineEdit, QPushButton, QFileDialog, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import torch
# from matplotlib.figure import Figure

class calculateDifference:
    def __init__(self, tensor_a, tensor_b):
        self.tensor_a = tensor_a
        self.tensor_b = tensor_b

    def l1_loss(self):
        l1_loss = torch.abs(self.tensor_a - self.tensor_b)
        return l1_loss.numpy()

    def l2_loss(self):
        l2_loss = torch.square(self.tensor_a - self.tensor_b)
        return l2_loss.numpy()

    def relative_error(self):
        relative_error = torch.abs(self.tensor_a - self.tensor_b) / torch.abs(self.tensor_a) * 100
        return relative_error.numpy()

    def tensor_difference_dict(self):
        tensor_diff = self.tensor_a - self.tensor_b
        l1_loss = self.l1_loss()
        l2_loss = self.l2_loss()
        rel_error = self.relative_error()

        errors_dict = {
            "Tensor Difference": tensor_diff,
            "L1 Error": l1_loss,
            "L2 Error": l2_loss,
            "Relative Error": rel_error
        }

        return errors_dict
    
def get_2d_tensor(tensor1, tensor2, num1, num2):
    num_dims = tensor1.ndim
    indices = [slice(None)] * num_dims
    for i in range(num_dims):
        if i != num1 and i!= num2:  # Exclude the selected dimension
            indices[i] = 0
    tensor1_2d = tensor1[tuple(indices)]
    tensor2_2d = tensor2[tuple(indices)]
    return tensor1_2d, tensor2_2d
    
class CustomNavigationToolbar(NavigationToolbar):
    def __init__(self, canvas, window):
        super().__init__(canvas, window)
        self.window = window

    def home(self):
        self.window.reset_graph()
        super().home()

class HeatmapWindow(QWidget):
    def __init__(self, tensor1, tensor2):
        super().__init__()
        self.setWindowTitle("Heatmap Window")
        self.tensor1 = tensor1
        self.tensor2 = tensor2
        self.errors_dict = dict()
        
        self.checkboxes = []
        self.selected_dimensions = []
        self.dimension_layout = QVBoxLayout()
        if self.tensor1.ndim == 1 or self.tensor1.ndim == 2:
            self.errors_dict = calculateDifference(self.tensor1, self.tensor2).tensor_difference_dict()
        else:
            # Create checkboxes for dimension selection
            # self.checkboxes = []
            # self.selected_dimensions = []
            print("3d tensor ")
            # self.dimension_layout = QVBoxLayout()
            self.create_dimension_checkboxes()
            # for i in range(self.tensor1.ndim):
            #     checkbox = QCheckBox(f"Dimension {i}")
            #     checkbox.stateChanged.connect(self.update_selected_dimensions)
            #     self.checkboxes.append(checkbox)
            # # Create a button to trigger reshaping
            self.dimen_button = QPushButton("Reshape")
            self.dimen_button.clicked.connect(self.reshape_tensor) # connect to regraph base on the shape selected
            # dimension_layout = QVBoxLayout()
            # for checkbox in self.checkboxes:
            #     dimension_layout.addWidget(checkbox)
            self.dimension_layout.addWidget(self.dimen_button)
            self.label = QLabel("")
            self.dimension_layout.addWidget(self.label)
                
        # self.figure = Figure()
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        self.dropdown = QComboBox(self)
        self.dropdown.addItems(["Tensor Difference", "L1 Error", "L2 Error", "Relative Error"])
        # self.dropdown.addItems(self.errors_dict.keys())
        self.dropdown.currentTextChanged.connect(self.draw_heatmap)

        self.color_button = QPushButton("Change Color")
        self.color_button.clicked.connect(self.change_color)
        
        # To Do check if there is a layout exit or not
        layout = QVBoxLayout()
        
        if self.tensor1.ndim != 1 or self.tensor1.ndim != 2:
            layout.addLayout(self.dimension_layout)
        layout.addWidget(self.dropdown)
        layout.addWidget(self.color_button)

        button_layout = QVBoxLayout()
        button_layout.addWidget(self.canvas)
        
        # Create the toolbar and add buttons
        # toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar = CustomNavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)

        # button_layout = QHBoxLayout()
        # button_layout.addWidget(self.dropdown)
        # button_layout.addWidget(self.color_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

        self.axes = None
        self.draw_heatmap(self.dropdown.currentText())

    def create_dimension_checkboxes(self):
        for i in range(self.tensor1.ndim):
            checkbox = QCheckBox(f"Dimension {i}")
            checkbox.stateChanged.connect(self.update_selected_dimensions)
            self.dimension_layout.addWidget(checkbox)
    
    def update_selected_dimensions(self):
        self.selected_dimensions = [i for i, checkbox in enumerate(self.dimension_layout.children()) if checkbox.isChecked()]
    
    def reshape_tensor(self):
        if len(self.selected_dimensions) != 2:
            self.label.setText("Please select exactly 2 dimensions.")
            return
        self.label.setText(f"Reshaped Tensor Shape: {self.selected_dimensions}")
        tensor1_2d, tensor2_2d= self.get_2d_dimension(self.selected_dimensions[0], self.selected_dimensions[1])
        self.errors_dict = calculateDifference(self.tensor1_2d, self.tensor2_2d).tensor_difference_dict()
        # reshaped_tensor = np.reshape(self.tensor, (self.tensor.shape[self.selected_dimensions[0]],
        #                                             self.tensor.shape[self.selected_dimensions[1]], -1))
        # reshaped_tensor = reshaped_tensor.reshape(reshaped_tensor.shape[0], reshaped_tensor.shape[1] * reshaped_tensor.shape[2])

        
        self.canvas.draw()
    
    def get_2d_dimension(self, dimension_x, dimension_y):
        return self.tensor1, self.tensor2
        
    # def update_selected_dimensions(self):
    #     self.selected_dimensions = [i for i, checkbox in enumerate(self.checkboxes) if checkbox.isChecked()]
    
    def draw_heatmap(self, text):
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        try:
            heatmap_data = self.errors_dict[text]
            if heatmap_data.shape[0] > 0 and heatmap_data.shape[1] > 0:
                self.heatmap = self.axes.imshow(heatmap_data, cmap='coolwarm', interpolation='nearest', aspect=1)
            else:
                print("Invalid heatmap data shape:", heatmap_data.shape)
                return
        except Exception as e:
            print("Error creating heatmap:", str(e))
            return

        # Create the mplcursors cursor
        cursor = mplcursors.cursor(hover=True)

        # Define the annotation function
        @cursor.connect("add")
        def on_hover(sel):
            x, y = sel.target.index
            value_tensor_diff = self.errors_dict["Tensor Difference"][x, y]
            value_l1 = self.errors_dict["L1 Error"][x, y]
            value_l2 = self.errors_dict["L2 Error"][x, y]
            value_relative = self.errors_dict["Relative Error"][x, y]
            sel.annotation.set_text(f"Tensor Location:({int(x)}, {int(y)})\nTensor Difference:{value_tensor_diff:.20f}\nL1 Error:{value_l1:.20f}\nL2 Error:{value_l2:.20f}\nRelative Error:{value_relative:.20f}")
    
        self.figure.colorbar(self.heatmap)
        
        # self.axes.set_box_aspect(1)
        # self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # self.toolbar.update()  # Update the toolbar 
        # TODO check if still need this, no self.toolbar in the Qwidget
        
        # Connect the zoom event to the canvas
        self.canvas.mpl_connect('scroll_event', self.zoom_heatmap)
        self.canvas.draw()

    def reset_graph(self):
        # self.figure.delaxes(self.axes)
        self.figure.clear()
        self.draw_heatmap(self.dropdown.currentText())  # Call draw_heatmap with the current selected text
    
    def change_color(self):
        # Update the colormap of the heatmap
        colormap = 'hot' if self.heatmap.get_cmap().name == 'coolwarm' else 'coolwarm'
        self.heatmap.set_cmap(colormap)
        self.canvas.draw()

    def zoom_heatmap(self, event):
        # Get the current x and y limits of the heatmap
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()

        # Compute the new limits based on the scroll event
        if event.xdata is not None and event.ydata is not None:
            scale_factor = 1.5 if event.button == 'up' else 1/1.5
            new_xlim = (
                (xlim[0] - event.xdata) * scale_factor + event.xdata,
                (xlim[1] - event.xdata) * scale_factor + event.xdata
            )
            new_ylim = (
                (ylim[0] - event.ydata) * scale_factor + event.ydata,
                (ylim[1] - event.ydata) * scale_factor + event.ydata
            )

            # Set the new limits to the heatmap
            self.axes.set_xlim(new_xlim)
            self.axes.set_ylim(new_ylim)

            # Redraw the heatmap
            self.canvas.draw()

class MainWindow(QMainWindow):
    def __init__(self, file1_path=None, file2_path=None):
        super().__init__()
        self.setWindowTitle("File Value Extractor")
        # self.errors_dict = dict()

        self.file1_label = QLabel("File 1:")
        self.file1_lineedit = QLineEdit()
        self.file1_button = QPushButton("Browse")

        self.file2_label = QLabel("File 2:")
        self.file2_lineedit = QLineEdit()
        self.file2_button = QPushButton("Browse")

        self.extract_button = QPushButton("Extract Values")

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
        self.layout.addWidget(self.extract_button)

        central_widget = QWidget()
        central_widget.setLayout(self.layout)
        self.setCentralWidget(central_widget)

        self.heatmap_window = None

        self.file1_button.clicked.connect(self.browse_file1)
        self.file2_button.clicked.connect(self.browse_file2)
        self.extract_button.clicked.connect(self.extract_values)

        if file1_path:
            self.file1_lineedit.setText(file1_path)
        if file2_path:
            self.file2_lineedit.setText(file2_path)

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
        if file1_path == file2_path:
            QMessageBox.critical(self, "Error", "Two files cannot be the same.")
            return
        try:
            data1 = np.load(file1_path)
            data2 = np.load(file2_path)

            if data1.shape != data2.shape:
                QMessageBox.critical(self, "Error", "Tensor sizes are not the same.")
                return

            tensor1 = torch.from_numpy(data1)
            tensor2 = torch.from_numpy(data2)

            if self.heatmap_window is not None:
                self.heatmap_window.close()
            
            
            self.heatmap_window = HeatmapWindow(tensor1, tensor2)
            self.heatmap_window.show()
        except FileNotFoundError:
            QMessageBox.critical(self, "Error", "File not found.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
