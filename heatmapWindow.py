import mplcursors
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QGroupBox, QComboBox, QSizePolicy, QWidget, QCheckBox, QFormLayout,  QVBoxLayout, QPushButton, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import torch
from matplotlib.figure import Figure

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


class Heatmap2DimenWindow(QWidget):
    def __init__(self, tensor1, tensor2):
        super().__init__()
        self.setWindowTitle("2D Heatmap Window")
        self.tensor1 = tensor1
        self.tensor2 = tensor2
        self.errors_dict = {}
        
        self.errors_dict = calculateDifference(self.tensor1, self.tensor2).tensor_difference_dict()
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        
        self.dropdown = QComboBox(self)
        self.dropdown.addItems(["Tensor Difference", "L1 Error", "L2 Error", "Relative Error"])
        self.dropdown.currentTextChanged.connect(self.draw_heatmap)
        
        self.color_button = QPushButton("Change Color")
        self.color_button.clicked.connect(self.change_color)
        
        layout = QVBoxLayout()
        layout.addWidget(self.dropdown)
        layout.addWidget(self.color_button)
        
        button_layout = QVBoxLayout()
        button_layout.addWidget(self.canvas)
        
        self.toolbar = CustomNavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
        self.axes = None
        self.draw_heatmap(self.dropdown.currentText())
        self.showMaximized()
    
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
            print("Error creating heatmap:", e)
            return
        
        # Create the mplcursors cursor
        cursor = mplcursors.cursor(hover=True)
        
        # Define the annotation function
        @cursor.connect("add")
        def on_hover(sel):
            x, y = sel.index
            value_tensor_diff = self.errors_dict["Tensor Difference"][x, y]
            value_l1 = self.errors_dict["L1 Error"][x, y]
            value_l2 = self.errors_dict["L2 Error"][x, y]
            value_relative = self.errors_dict["Relative Error"][x, y]
            sel.annotation.set_text(f"Tensor Location:({int(x)}, {int(y)})\nTensor Difference:{value_tensor_diff:.20f}\nL1 Error:{value_l1:.20f}\nL2 Error:{value_l2:.20f}\nRelative Error:{value_relative:.20f}")
        
        self.figure.colorbar(self.heatmap)
        
        self.canvas.mpl_connect('scroll_event', self.zoom_heatmap)
        self.canvas.draw()
    
    def reset_graph(self):
        self.figure.clear()
        self.draw_heatmap(self.dropdown.currentText())  # Call draw_heatmap with the current selected text
    
    def change_color(self):
        # Update the colormap of the heatmap
        try:
            colormap = 'hot' if self.heatmap.get_cmap().name == 'coolwarm' else 'coolwarm'
            self.heatmap.set_cmap(colormap)
            self.canvas.draw()
        except Exception as e:
            QMessageBox.critical(self, "Error: Not able to change color: ", str(e))
            return
    
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


class Canvas(FigureCanvas):
    def __init__(self, parent=None):
        self.figure = Figure()
        super().__init__(self.figure)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.axes = self.figure.add_subplot(111)
        self.heatmap = None
    
    def clear_canvas(self):
        self.figure.clear()
        self.heatmap = None
        self.draw()
    
    def get_figure(self):
        return self.figure
    
    def draw_heatmap(self, text, errors_dict):
        self.clear_canvas()
        self.axes = self.figure.add_subplot(111)
        
        try:
            heatmap_data = errors_dict[text]
            if heatmap_data.shape[0] > 0 and heatmap_data.shape[1] > 0:
                self.heatmap = self.axes.imshow(heatmap_data, cmap='coolwarm', interpolation='nearest', aspect=1)
            else:
                print("Invalid heatmap data shape:", heatmap_data.shape)
                return
        except Exception as e:
            if len(errors_dict) == 0:
                return
            print("Error creating heatmap:", e)
            return
        
        cursor = mplcursors.cursor(self.heatmap, hover=True)
        
        @cursor.connect("add")
        def on_hover(sel):
            x, y = sel.index
            value_tensor_diff = errors_dict["Tensor Difference"][x, y]
            value_l1 = errors_dict["L1 Error"][x, y]
            value_l2 = errors_dict["L2 Error"][x, y]
            value_relative = errors_dict["Relative Error"][x, y]
            sel.annotation.set_text(f"Tensor Location:({int(x)}, {int(y)})\nTensor Difference:{value_tensor_diff:.20f}\nL1 Error:{value_l1:.20f}\nL2 Error:{value_l2:.20f}\nRelative Error:{value_relative:.20f}")
        
        self.figure.colorbar(self.heatmap, ax=self.axes)
        self.mpl_connect('scroll_event', self.zoom_heatmap)
        self.draw()
    
    def change_color(self):
        try:
            colormap = 'hot' if self.heatmap.get_cmap().name == 'coolwarm' else 'coolwarm'
            self.heatmap.set_cmap(colormap)
            self.draw()
        except Exception as e:
            QMessageBox.critical(self, "Error: Not able to change color: ", str(e))
            return
    
    def zoom_heatmap(self, event):
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        
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
            
            self.axes.set_xlim(new_xlim)
            self.axes.set_ylim(new_ylim)
            
            self.draw()

class HeatmapMultiDimenWindow(QWidget):
    def __init__(self, tensor1, tensor2):
        super().__init__()
        self.setWindowTitle("Multidimensional Heatmap Window")
        self.tensor1 = tensor1
        self.tensor2 = tensor2
        self.errors_dict = {}
        
        self.checkboxes = []
        self.dropdowns = []
        self.selected_checkboxes = []
        
        main_layout = QVBoxLayout(self)
        group_box = QGroupBox("Please select two dimensions to graph and fix the rest dimensions.")
        group_box_layout = QVBoxLayout(group_box)      
        self.create_widgets()
        select_dimen_layout=self.create_layout()
        
        self.dropdown = QComboBox(self)
        self.dropdown.addItems(["Tensor Difference", "L1 Error", "L2 Error", "Relative Error"])
        
        self.color_button = QPushButton("Change Color")
        
        self.canvas = Canvas()
        
        select_dimen_layout.addRow(self.dropdown, self.color_button)
        
        self.toolbar = CustomNavigationToolbar(self.canvas, self)
        select_dimen_layout.addRow(self.toolbar)
        
        group_box_layout.addLayout(select_dimen_layout)
        group_box_layout.addWidget(self.canvas)
        
        self.dropdown.currentTextChanged.connect(lambda text: self.canvas.draw_heatmap(text, self.errors_dict))
        self.color_button.clicked.connect(self.canvas.change_color)
        
        main_layout.addWidget(group_box)
        self.showMaximized()
    
    def slice_2d_tensor(self, selected_dimensions, fixed_dimensions): # selected_dimensions, fixed_dimensions
        num_dims = self.tensor1.ndim
        indices = [slice(None)] * num_dims
        for i in range(num_dims):
            if i not in selected_dimensions: 
                indices[i] = fixed_dimensions[i]
                # indices[i] = 0
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
            dropdown.setCurrentIndex(0)  # Set default value
            self.dropdowns.append(dropdown)
        
        self.graph_button = QPushButton("Graph")
        self.graph_button.clicked.connect(self.graph_button_clicked)
        
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_button_clicked)
    
    def create_layout(self):
        layout = QFormLayout()
        for checkbox, dropdown in zip(self.checkboxes, self.dropdowns):
            layout.addRow(checkbox, dropdown)
        layout.addRow(self.reset_button, self.graph_button)
        # self.setLayout(layout)
        return layout
    
    def checkbox_changed(self, checkbox):
        if checkbox.isChecked():
            if len(self.selected_checkboxes) >= 2:
                checkbox.setChecked(False)  # Prevent selecting more than two checkboxes
            else:
                self.selected_checkboxes.append(checkbox)
                dropdown = self.dropdowns[self.checkboxes.index(checkbox)]
                dropdown.setCurrentIndex(0)  # Set dropdown to default value
                dropdown.setEnabled(False)  # Disable the associated dropdown
        else:
            if checkbox in self.selected_checkboxes:
                self.selected_checkboxes.remove(checkbox)
            dropdown = self.dropdowns[self.checkboxes.index(checkbox)]
            dropdown.setEnabled(True)  # Enable the associated dropdown
    
    def dropdown_changed(self, index, checkbox):
        if index != 0:  # Option 1 is not selected
            checkbox.setChecked(False)  # Uncheck the associated checkbox
    
    def graph_button_clicked(self):
        if len(self.selected_checkboxes) == 2:
            selected_dimensions = []
            for checkbox in self.selected_checkboxes:
                dimension_index = self.checkboxes.index(checkbox)
                selected_dimensions.append(dimension_index)
            
            all_dimensions = list(range(len(self.checkboxes)))
            all_axes = [self.dropdowns[i].currentIndex() for i in range(len(self.checkboxes))]
            
            remaining_dimensions = [dim for dim in all_dimensions if dim not in selected_dimensions]
            remaining_axes = [all_axes[dim] for dim in remaining_dimensions]
            fixed_dimensions = dict(zip(remaining_dimensions, remaining_axes))
            print("selected_dimensions, fixed_dimensions", selected_dimensions, fixed_dimensions)
            tensor1_2d, tensor2_2d= self.slice_2d_tensor(selected_dimensions, fixed_dimensions)
            print("tensor1_2d", tensor1_2d.shape)
            self.errors_dict = calculateDifference(tensor1_2d, tensor2_2d).tensor_difference_dict()
            self.canvas.draw_heatmap(self.dropdown.currentText(), self.errors_dict)
        else:
            QMessageBox.warning(self, "Graph", "Please select exactly two checkboxes.")
    
    def reset_button_clicked(self):
        for checkbox in self.checkboxes:
            checkbox.setChecked(False)
        for dropdown in self.dropdowns:
            dropdown.setEnabled(True)
            dropdown.setCurrentIndex(0)
        self.selected_checkboxes.clear()
        self.errors_dict = {}
        self.canvas.clear_canvas()
    
    def reset_graph(self):
        self.canvas.clear_canvas()
        self.canvas.draw_heatmap(self.dropdown.currentText(), self.errors_dict)  # Call draw_heatmap with the current selected text
