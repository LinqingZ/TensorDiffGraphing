import numpy as np
import mplcursors
import matplotlib
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QSlider, QGroupBox, QComboBox, QSizePolicy, QWidget, QCheckBox, QFormLayout, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import torch
from PyQt6.QtCore import Qt
from matplotlib.figure import Figure

class My_Axes(matplotlib.axes.Axes):
    name = "My_Axes"
    def drag_pan(self, button, key, x, y):
        matplotlib.axes.Axes.drag_pan(self, button, 'x', x, y) # pretend key=='x'

matplotlib.projections.register_projection(My_Axes)

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


class CustomNavigationToolbar(NavigationToolbar):
    def __init__(self, canvas, window):
        super().__init__(canvas, window)
        self.window = window
    
    def home(self):
        self.window.reset_graph()
        super().home()


class Histogram2DimenWindow(QWidget):
    def __init__(self, tensor1, tensor2):
        super().__init__()
        self.setWindowTitle("2D Histogram Window")
        self.tensor1 = tensor1
        self.tensor2 = tensor2
        self.errors_dict = {}
        self.bins = None
        self.axes = None
        self.bin_size = None
        self.log_base = None
        self.factor = 1
        
        self.errors_dict = calculateDifference(self.tensor1, self.tensor2).tensor_difference_dict()
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(100)
        self.slider.setTickInterval(1)
        self.slider.setSingleStep(1)
        self.slider.setValue(10)
        self.slider.valueChanged[int].connect(self.update_bin_size)
        
        self.display_message_label = QLabel("Bin Size:")
        self.bin_size_textbox = QLineEdit('10')
        self.bin_size_textbox.textChanged.connect(self.update_bin_size_textbox)
        
        self.plot_button = QPushButton('Plot Histogram')
        self.plot_button.clicked.connect(self.draw_histogram)
        
        self.dropdown = QComboBox(self)
        self.dropdown.addItems(["Tensor Difference", "L1 Error", "L2 Error", "Relative Error"])
        
        self.log_checkbox = QCheckBox('Logarithmic Scale Base 10')
        self.log_checkbox.stateChanged.connect(self.update_log_base)
        
        main_layout = QVBoxLayout(self)
        group_box = QGroupBox("Bin Size")
        layout = QVBoxLayout(group_box)
        layout.addWidget(self.slider)
        layout.addWidget(self.bin_size_textbox)
        layout.addWidget(self.dropdown)
        layout.addWidget(self.log_checkbox)
        layout.addWidget(self.plot_button)
        
        button_layout = QVBoxLayout()
        button_layout.addWidget(self.canvas)
        
        self.toolbar = CustomNavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        
        layout.addLayout(button_layout)
        main_layout.addWidget(group_box)
        # self.setLayout(layout1)
        
        self.axes = None
        self.draw_histogram()
        self.showMaximized()
    
    def update_log_base(self):
        if self.log_checkbox.isChecked():
            self.log_base = 10
        else:
            self.log_base = None
    
    def update_bin_size(self, bin_size):
        self.bin_size_textbox.setText(str(bin_size))
    
    def update_bin_size_textbox(self):
        bin_size_text = self.bin_size_textbox.text()
        if bin_size_text.isdigit():
            self.bin_size = int(bin_size_text)
            self.slider.setValue(self.bin_size)
    
    def zoom_graph(self, event):
        xlim = self.axes.get_xlim()
        if event.xdata is not None and event.ydata is not None:
            scale_factor = 1.5 if event.button == 'up' else 1/1.5
            new_xlim = (
                (xlim[0] - event.xdata) * scale_factor + event.xdata,
                (xlim[1] - event.xdata) * scale_factor + event.xdata
            )

            self.axes.set_xlim(new_xlim)
            self.canvas.draw()
    
    def draw_histogram(self):
        self.figure.clear()
        self.bin_size = int(self.bin_size_textbox.text())
        self.axes = self.figure.add_subplot(111, projection="My_Axes")
        if self.log_base != None:
            self.axes.set_yscale('log', base=self.log_base)
        frequencies, self.bins, patches = self.axes.hist(self.errors_dict[self.dropdown.currentText()].flatten(), bins=self.bin_size)
        
        self.axes.set_xlabel('Value')
        self.axes.set_ylabel('Frequency')
        self.axes.set_title('Histogram')
        
        # Assign different colors to each bin
        cmap = plt.get_cmap('viridis')
        bin_colors = cmap(np.linspace(0, 1, len(patches)))
        for patch, color in zip(patches, bin_colors):
            patch.set_facecolor(color)
        
        # Create the mplcursors cursor
        cursor = mplcursors.cursor(patches, hover=True)
        # Define the annotation function
        @cursor.connect("add")
        def on_hover(sel):
            sel.annotation.set_text(f"Range: [{sel.artist[sel.index].get_x()}, {sel.artist[sel.index].get_x() + sel.artist[sel.index].get_width()}]\nFrequency: {int(frequencies[sel.index])}")
        min_bin = min(self.bins)
        max_bin = max(self.bins)
        self.axes.set(xlim=(min_bin, max_bin), ylim=(0, None), autoscale_on=False)
        
        self.canvas.mpl_connect('scroll_event', self.zoom_graph)
        self.canvas.draw()
    
    def reset_graph(self):
        self.figure.clear()
        self.draw_histogram()


class Canvas(FigureCanvas):
    def __init__(self, parent=None):
        self.figure = Figure()
        super().__init__(self.figure)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.axes = self.figure.add_subplot(111)
        self.histogram = None
    
    def clear_canvas(self):
        self.figure.clear()
        self.histogram = None
        self.draw()
    
    def get_figure(self):
        return self.figure
    
    def draw_histogram(self, text, errors_dict, bin_size, log_base):
        self.clear_canvas()
        self.axes = self.figure.add_subplot(111, projection="My_Axes")
        if log_base != None:
            self.axes.set_yscale('log', base=log_base)
        frequencies, bins, patches = self.axes.hist(errors_dict[text].flatten(), bins=bin_size)
        
        self.axes.set_xlabel('Value')
        self.axes.set_ylabel('Frequency')
        self.axes.set_title('Histogram')
        
        # Assign different colors to each bin
        cmap = plt.get_cmap('viridis')
        bin_colors = cmap(np.linspace(0, 1, len(patches)))
        for patch, color in zip(patches, bin_colors):
            patch.set_facecolor(color)
        
        # Create the mplcursors cursor
        cursor = mplcursors.cursor(patches, hover=True)
        # Define the annotation function
        @cursor.connect("add")
        def on_hover(sel):
            sel.annotation.set_text(f"Range: [{sel.artist[sel.index].get_x()}, {sel.artist[sel.index].get_x() + sel.artist[sel.index].get_width()}]\nFrequency: {int(frequencies[sel.index])}")
        
        min_bin = min(bins)
        max_bin = max(bins)
        self.axes.set(xlim=(min_bin, max_bin), ylim=(0, None), autoscale_on=False)
        
        self.mpl_connect('scroll_event', self.zoom_graph)
        self.draw()
    
    def zoom_graph(self, event):
        xlim = self.axes.get_xlim()
        if event.xdata is not None and event.ydata is not None:
            scale_factor = 1.5 if event.button == 'up' else 1/1.5
            new_xlim = (
                (xlim[0] - event.xdata) * scale_factor + event.xdata,
                (xlim[1] - event.xdata) * scale_factor + event.xdata
            )
            
            self.axes.set_xlim(new_xlim)
            self.draw()

class HistogramMultiDimenWindow(QWidget):
    def __init__(self, tensor1, tensor2):
        super().__init__()
        self.setWindowTitle("Multidimensional Histogram Window")
        self.tensor1 = tensor1
        self.tensor2 = tensor2
        self.bin_size = None
        self.log_base = None
        self.errors_dict = {}
        self.checkboxes = []
        self.axis_dropdowns = []
        self.selected_checkboxes = []
        
        self.canvas = Canvas()
        main_layout = QVBoxLayout(self)
        group_box = QGroupBox("Please select two dimensions to graph and fix the rest dimensions.")
        group_box_layout = QVBoxLayout(group_box)      
        self.create_widgets()
        select_dimen_layout=self.create_layout()
        
        group_box_layout.addLayout(select_dimen_layout)
        group_box_layout.addWidget(self.canvas)
        
        main_layout.addWidget(group_box)
        self.showMaximized()
    
    def slice_2d_tensor(self, selected_dimensions, fixed_dimensions):
        num_dims = self.tensor1.ndim
        indices = [slice(None)] * num_dims
        for i in range(num_dims):
            if i not in selected_dimensions: 
                indices[i] = fixed_dimensions[i]
        print("selected_dimensions", selected_dimensions)
        print("fixed_dimensions", fixed_dimensions)
        print("indices", indices)
        tensor1_2d = self.tensor1[tuple(indices)]
        tensor2_2d = self.tensor2[tuple(indices)]
        return tensor1_2d, tensor2_2d
    
    def create_widgets(self):
        tensor_shape = self.tensor1.shape
        
        for i in range(len(tensor_shape)):
            checkbox = QCheckBox(f"Dimension {i}")
            checkbox.stateChanged.connect(lambda state, checkbox=checkbox: self.checkbox_changed(checkbox))
            self.checkboxes.append(checkbox)
            
            dropdown = QComboBox()
            for j in range(tensor_shape[i]):
                dropdown.addItem(f"Axis {j}")
            dropdown.currentIndexChanged.connect(lambda index, checkbox=checkbox: self.dropdown_changed(index, checkbox))
            dropdown.setCurrentIndex(0)
            self.axis_dropdowns.append(dropdown)
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(100)
        self.slider.setTickInterval(1)
        self.slider.setSingleStep(1)
        self.slider.setValue(10)
        self.slider.valueChanged[int].connect(self.update_bin_size)
        
        self.bin_size_textbox = QLineEdit('10')
        self.bin_size_textbox.textChanged.connect(self.update_bin_size_textbox)
        
        self.graph_button = QPushButton("Plot Histogram")
        self.graph_button.clicked.connect(self.graph_button_clicked)
        
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_button_clicked)
        
        self.dropdown = QComboBox(self)
        self.dropdown.addItems(["Tensor Difference", "L1 Error", "L2 Error", "Relative Error"])
        
        self.log_checkbox = QCheckBox('Logarithmic Scale Base 10')
        self.log_checkbox.stateChanged.connect(self.update_log_base)
        
        self.toolbar = CustomNavigationToolbar(self.canvas, self)
    
    def create_layout(self):
        layout = QFormLayout()
        for checkbox, dropdown in zip(self.checkboxes, self.axis_dropdowns):
            layout.addRow(checkbox, dropdown)
        layout.addRow(self.slider)
        layout.addRow(self.bin_size_textbox)
        layout.addRow(self.log_checkbox)
        layout.addRow(self.reset_button, self.dropdown)
        layout.addRow(self.graph_button)
        layout.addRow(self.toolbar)
        return layout
    
    def update_log_base(self):
        if self.log_checkbox.isChecked():
            self.log_base = 10
        else:
            self.log_base = None
    
    def update_bin_size(self, bin_size):
        self.bin_size_textbox.setText(str(bin_size))
    
    def update_bin_size_textbox(self):
        bin_size_text = self.bin_size_textbox.text()
        if bin_size_text.isdigit():
            self.bin_size = int(bin_size_text)
            self.slider.setValue(self.bin_size)
    
    def checkbox_changed(self, checkbox):
        if checkbox.isChecked():
            if len(self.selected_checkboxes) >= 2:
                checkbox.setChecked(False)  # Prevent selecting more than two checkboxes
            else:
                self.selected_checkboxes.append(checkbox)
                dropdown = self.axis_dropdowns[self.checkboxes.index(checkbox)]
                dropdown.setCurrentIndex(0)  # Set dropdown to default value
                dropdown.setEnabled(False)  # Disable the associated dropdown
        else:
            if checkbox in self.selected_checkboxes:
                self.selected_checkboxes.remove(checkbox)
            dropdown = self.axis_dropdowns[self.checkboxes.index(checkbox)]
            dropdown.setEnabled(True)  # Enable the associated dropdown
    
    def dropdown_changed(self, index, checkbox):
        if index != 0:
            checkbox.setChecked(False)  # Uncheck the associated checkbox
    
    def graph_button_clicked(self):
        if len(self.selected_checkboxes) == 2:
            selected_dimensions = []
            for checkbox in self.selected_checkboxes:
                dimension_index = self.checkboxes.index(checkbox)
                selected_dimensions.append(dimension_index)
            
            all_dimensions = list(range(len(self.checkboxes)))
            all_axes = [self.axis_dropdowns[i].currentIndex() for i in range(len(self.checkboxes))]
            
            remaining_dimensions = [dim for dim in all_dimensions if dim not in selected_dimensions]
            remaining_axes = [all_axes[dim] for dim in remaining_dimensions]
            fixed_dimensions = dict(zip(remaining_dimensions, remaining_axes))
            print("selected_dimensions, fixed_dimensions", selected_dimensions, fixed_dimensions)
            tensor1_2d, tensor2_2d= self.slice_2d_tensor(selected_dimensions, fixed_dimensions)
            print("tensor1_2d", tensor1_2d.shape)
            self.errors_dict = calculateDifference(tensor1_2d, tensor2_2d).tensor_difference_dict()
            self.canvas.draw_histogram(self.dropdown.currentText(), self.errors_dict, self.bin_size, self.log_base)
        else:
            QMessageBox.warning(self, "Graph", "Please select exactly two checkboxes.")
    
    def reset_button_clicked(self):
        for checkbox in self.checkboxes:
            checkbox.setChecked(False)
        for dropdown in self.axis_dropdowns:
            dropdown.setEnabled(True)
            dropdown.setCurrentIndex(0)
        self.selected_checkboxes.clear()
        self.errors_dict = {}
        self.slider.setValue(10)
        self.canvas.clear_canvas()
    
    def reset_graph(self):
        self.canvas.clear_canvas()
        self.canvas.draw_histogram(self.dropdown.currentText(), self.errors_dict, self.bin_size)  # Call draw_histogram with the current selected text
