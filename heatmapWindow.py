import mplcursors
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QGroupBox, QComboBox, QSizePolicy, QWidget, QCheckBox, QLabel, QFormLayout,  QVBoxLayout, QHBoxLayout, QPushButton, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import torch
import numpy as np
from matplotlib.colors import LogNorm
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
        tensor_diff = (self.tensor_a - self.tensor_b).numpy()
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
        self.original_xlim = None
        self.original_ylim = None
        
        self.errors_dict = calculateDifference(self.tensor1, self.tensor2).tensor_difference_dict()
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        
        self.dropdown = QComboBox(self)
        self.dropdown.addItems(["Tensor Difference", "L1 Error", "L2 Error", "Relative Error"])
        self.dropdown.currentTextChanged.connect(self.draw_heatmap)
        
        self.color_button = QPushButton("Change Color")
        self.color_button.clicked.connect(self.change_color)
        
        self.color_scale_checkbox = QCheckBox('Scale Color')
        self.color_scale_checkbox.stateChanged.connect(self.scale_color)
        
        main_layout = QVBoxLayout(self)
        
        group_box = QGroupBox()
        inner_layout = QHBoxLayout(group_box)
        left_layout = QFormLayout()
        left_layout.addRow(self.dropdown)
        left_layout.addRow(self.color_button)
        left_layout.addRow(self.color_scale_checkbox)
        
        
        self.mean_label = QLabel()
        self.median_label = QLabel()
        self.max_label = QLabel()
        self.min_label = QLabel()
        self.std_label = QLabel()
        self.percentiles_label_25 = QLabel()
        self.percentiles_label_50 = QLabel()
        self.percentiles_label_75 = QLabel()
        
        left_layout.addRow(QLabel("<b>Min: </b>"))
        left_layout.addRow(self.min_label)
        
        left_layout.addRow(QLabel("<b>Max: </b>"))
        left_layout.addRow(self.max_label)
        
        left_layout.addRow(QLabel("<b>Median: </b>"))
        left_layout.addRow(self.median_label)
        
        left_layout.addRow(QLabel("<b>Mean (Average): </b>"))
        left_layout.addRow(self.mean_label)
        
        left_layout.addRow(QLabel("<b>Standard Deviation (SD): </b>"))
        left_layout.addRow(self.std_label)
        
        left_layout.addRow(QLabel("<b>Percentiles 25th: </b>"))
        left_layout.addRow(self.percentiles_label_25)
        
        left_layout.addRow(QLabel("<b>Percentiles 50th: </b>"))
        left_layout.addRow(self.percentiles_label_50)
        
        left_layout.addRow(QLabel("<b>Percentiles 75th: </b>"))
        left_layout.addRow(self.percentiles_label_75)
        
        
        right_layout = QVBoxLayout()
        self.toolbar = CustomNavigationToolbar(self.canvas, self)
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)
        
        inner_layout.addLayout(left_layout, 1)
        inner_layout.addLayout(right_layout, 7)
        main_layout.addWidget(group_box)
        # self.setLayout(group_box)
        
        self.axes = None
        self.draw_heatmap(self.dropdown.currentText())
        self.showMaximized()
    
    def scale_color(self):
        if self.color_scale_checkbox.isChecked:
            self.draw_heatmap(self.dropdown.currentText())
    
    def update_statistics(self, data_tensor):
        mean_value = np.mean(data_tensor)
        median_value = np.median(data_tensor)
        max_value = np.max(data_tensor)
        # max_location = np.unravel_index(np.argmax(data_tensor), data_tensor.shape)
        min_value = np.min(data_tensor)
        # min_location = np.unravel_index(np.argmin(data_tensor), data_tensor.shape)
        std_deviation = np.std(data_tensor)
        # variance = np.var(data_tensor)
        # percentiles = np.percentile(data_tensor, [25, 50, 75])
        percentiles_25 = np.percentile(data_tensor, 25)
        percentiles_50 = np.percentile(data_tensor, 50)
        percentiles_75 = np.percentile(data_tensor, 75)

        self.mean_label.setText(f"{mean_value:.10e}")
        self.median_label.setText(f"{median_value:.10e}")
        self.max_label.setText(f"{max_value:.10e}")
        self.min_label.setText(f"{min_value:.10e}")
        self.std_label.setText(f"{std_deviation:.10e}")
        self.percentiles_label_25.setText(f"{percentiles_25:.10e}")
        self.percentiles_label_50.setText(f"{percentiles_50:.10e}")
        self.percentiles_label_75.setText(f"{percentiles_75:.10e}")

    def draw_heatmap(self, text):
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        try:
            heatmap_data = self.errors_dict[text]
            self.update_statistics(heatmap_data)
            if heatmap_data.shape[0] > 0 and heatmap_data.shape[1] > 0:
                if self.color_scale_checkbox.isChecked():
                    self.heatmap = self.axes.imshow(heatmap_data, cmap='coolwarm', interpolation='nearest', aspect=1, norm=LogNorm())
                else:
                    self.heatmap = self.axes.imshow(heatmap_data, cmap='coolwarm', interpolation='nearest', aspect=1)
                # Store the original heatmap data's range
                if self.original_xlim is None:
                    self.original_xlim = self.axes.get_xlim()
                    self.original_ylim = self.axes.get_ylim()
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
            sel.annotation.set_text(f"Tensor Location: ({int(x)}, {int(y)})\nTensor 1: {self.tensor1[x, y]}\nTensor 2: {self.tensor2[x, y]}\nTensor Difference: {value_tensor_diff:.20f}\nL1 Error: {value_l1:.20f}\nL2 Error: {value_l2:.20f}\nRelative Error: {value_relative:.20f}")
        
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
            
            # # Check if the new limits exceed the original limits and adjust if needed
            # if new_xlim[0] < self.original_xlim[0] or new_xlim[1] > self.original_xlim[1]:
            #     new_xlim = self.original_xlim
            # if new_ylim[0] > self.original_ylim[0] or new_ylim[1] < self.original_ylim[1]:
            #     new_ylim = self.original_ylim
            new_xlim = (max(self.original_xlim[0], new_xlim[0]), min(self.original_xlim[1], new_xlim[1]))
            new_ylim = (min(self.original_ylim[0], new_ylim[0]), max(self.original_ylim[1], new_ylim[1]))
            
            # Set the new limits to the heatmap
            self.axes.set_xlim((new_xlim[0], new_xlim[1]))
            self.axes.set_ylim((new_ylim[0], new_ylim[1]))
            
            # Redraw the heatmap
            self.canvas.draw()


class Canvas(FigureCanvas):
    def __init__(self, parent=None):
        self.figure = Figure()
        super().__init__(self.figure)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.axes = self.figure.add_subplot(111)
        self.heatmap = None
        self.original_xlim = None
        self.original_ylim = None
        self.scale_color = False
    
    def change_color_scale(self, value):
        self.scale_color = value
    
    def clear_canvas(self):
        self.figure.clear()
        self.heatmap = None
        self.draw()
    
    def get_figure(self):
        return self.figure
    
    def draw_heatmap(self, text, errors_dict, tensor1, tensor2):
        self.clear_canvas()
        self.axes = self.figure.add_subplot(111)
        
        try:
            heatmap_data = errors_dict[text]
            if heatmap_data.shape[0] > 0 and heatmap_data.shape[1] > 0:
                if self.scale_color:
                    self.heatmap = self.axes.imshow(heatmap_data, cmap='coolwarm', interpolation='nearest', aspect=1, norm=LogNorm())
                else:
                    self.heatmap = self.axes.imshow(heatmap_data, cmap='coolwarm', interpolation='nearest', aspect=1)

                # Store the original heatmap data's range
                if self.original_xlim is None:
                    self.original_xlim = self.axes.get_xlim()
                    self.original_ylim = self.axes.get_ylim()
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
            sel.annotation.set_text(f"Tensor Location: ({int(x)}, {int(y)})\nTensor 1: {tensor1[x, y]}\nTensor 2: {tensor2[x, y]}\nTensor Difference: {value_tensor_diff:.20f}\nL1 Error: {value_l1:.20f}\nL2 Error: {value_l2:.20f}\nRelative Error: {value_relative:.20f}")
        
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
            
            new_xlim = (max(self.original_xlim[0], new_xlim[0]), min(self.original_xlim[1], new_xlim[1]))
            new_ylim = (min(self.original_ylim[0], new_ylim[0]), max(self.original_ylim[1], new_ylim[1]))
            
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
        self.tensor1_2d = None
        self.tensor2_2d = None
        
        self.checkboxes = []
        self.dropdowns = []
        self.selected_checkboxes = []
        
        main_layout = QVBoxLayout(self)
        group_box = QGroupBox("Please select two dimensions to graph and fix the rest dimensions.")
        group_box_layout = QHBoxLayout(group_box)      
        self.create_widgets()
        select_dimen_layout=self.create_layout()
        
        self.dropdown = QComboBox(self)
        self.dropdown.addItems(["Tensor Difference", "L1 Error", "L2 Error", "Relative Error"])
        
        self.color_button = QPushButton("Change Color")
        
        self.canvas = Canvas()
        
        select_dimen_layout.addRow(self.dropdown, self.color_button)
        
        self.scale_color_checkbox = QCheckBox('Scale Color')
        select_dimen_layout.addRow(self.scale_color_checkbox)
        
        self.mean_label = QLabel()
        self.median_label = QLabel()
        self.max_label = QLabel()
        self.min_label = QLabel()
        self.std_label = QLabel()
        self.percentiles_label_25 = QLabel()
        self.percentiles_label_50 = QLabel()
        self.percentiles_label_75 = QLabel()
        
        select_dimen_layout.addRow(QLabel("<b>Min: </b>"))
        select_dimen_layout.addRow(self.min_label)
        
        select_dimen_layout.addRow(QLabel("<b>Max: </b>"))
        select_dimen_layout.addRow(self.max_label)
        
        select_dimen_layout.addRow(QLabel("<b>Median: </b>"))
        select_dimen_layout.addRow(self.median_label)
        
        select_dimen_layout.addRow(QLabel("<b>Mean (Average): </b>"))
        select_dimen_layout.addRow(self.mean_label)
        
        select_dimen_layout.addRow(QLabel("<b>Standard Deviation (SD): </b>"))
        select_dimen_layout.addRow(self.std_label)
        
        select_dimen_layout.addRow(QLabel("<b>Percentiles 25th: </b>"))
        select_dimen_layout.addRow(self.percentiles_label_25)
        
        select_dimen_layout.addRow(QLabel("<b>Percentiles 50th: </b>"))
        select_dimen_layout.addRow(self.percentiles_label_50)
        
        select_dimen_layout.addRow(QLabel("<b>Percentiles 75th: </b>"))
        select_dimen_layout.addRow(self.percentiles_label_75)
        
        
        self.toolbar = CustomNavigationToolbar(self.canvas, self)
        
        group_box_layout.addLayout(select_dimen_layout)
        canvas_layout = QVBoxLayout()
        canvas_layout.addWidget(self.toolbar)
        canvas_layout.addWidget(self.canvas)
        group_box_layout.addLayout(canvas_layout)
        
        self.dropdown.currentTextChanged.connect(lambda text: self.canvas.draw_heatmap(text, self.errors_dict, self.tensor1_2d, self.tensor2_2d))
        self.dropdown.currentTextChanged.connect(lambda text: self.update_statistics(self.errors_dict[text]))
        self.color_button.clicked.connect(self.canvas.change_color)
        self.scale_color_checkbox.stateChanged.connect(self.change_color_scale)
        
        
        main_layout.addWidget(group_box)
        self.showMaximized()
    
    def update_statistics(self, data_tensor):
        mean_value = np.mean(data_tensor)
        median_value = np.median(data_tensor)
        max_value = np.max(data_tensor)
        # max_location = np.unravel_index(np.argmax(data_tensor), data_tensor.shape)
        min_value = np.min(data_tensor)
        # min_location = np.unravel_index(np.argmin(data_tensor), data_tensor.shape)
        std_deviation = np.std(data_tensor)
        # variance = np.var(data_tensor)
        # percentiles = np.percentile(data_tensor, [25, 50, 75])
        percentiles_25 = np.percentile(data_tensor, 25)
        percentiles_50 = np.percentile(data_tensor, 50)
        percentiles_75 = np.percentile(data_tensor, 75)

        self.mean_label.setText(f"{mean_value:.10e}")
        self.median_label.setText(f"{median_value:.10e}")
        self.max_label.setText(f"{max_value:.10e}")
        self.min_label.setText(f"{min_value:.10e}")
        self.std_label.setText(f"{std_deviation:.10e}")
        self.percentiles_label_25.setText(f"{percentiles_25:.10e}")
        self.percentiles_label_50.setText(f"{percentiles_50:.10e}")
        self.percentiles_label_75.setText(f"{percentiles_75:.10e}")
    
    def change_color_scale(self):
        if self.scale_color_checkbox.isChecked():
            self.canvas.change_color_scale(True)
        else:
            self.canvas.change_color_scale(False)
        self.canvas.draw_heatmap(self.dropdown.currentText(), self.errors_dict, self.tensor1_2d, self.tensor2_2d)
    
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
        self.tensor1_2d = self.tensor1[tuple(indices)]
        self.tensor2_2d = self.tensor2[tuple(indices)]
        # return tensor1_2d, tensor2_2d
    
    def create_widgets(self):
        tensor_shape = self.tensor1.shape
        
        for i in range(len(tensor_shape)):
            checkbox = QCheckBox(f"Dimension {i}")
            checkbox.stateChanged.connect(lambda state, checkbox=checkbox: self.checkbox_changed(checkbox))
            self.checkboxes.append(checkbox)
            
            dropdown = QComboBox()
            for j in range(tensor_shape[i]):
                dropdown.addItem(f"Index {j}")
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
        self.freeze_unselected_checkboxes()
    
    def freeze_unselected_checkboxes(self):
        if len(self.selected_checkboxes) >= 2:
            for checkbox in self.checkboxes:
                if checkbox not in self.selected_checkboxes:
                    checkbox.setEnabled(False)
                else:
                    checkbox.setEnabled(True)
        else:
            for checkbox in self.checkboxes:
                checkbox.setEnabled(True)
    
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
            # tensor1_2d, tensor2_2d= self.slice_2d_tensor(selected_dimensions, fixed_dimensions)
            self.slice_2d_tensor(selected_dimensions, fixed_dimensions)
            print("tensor1_2d", self.tensor1_2d.shape)
            self.errors_dict = calculateDifference(self.tensor1_2d, self.tensor2_2d).tensor_difference_dict()
            self.canvas.draw_heatmap(self.dropdown.currentText(), self.errors_dict, self.tensor1_2d, self.tensor2_2d)
            self.update_statistics(self.errors_dict[self.dropdown.currentText()])
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
        self.canvas.draw_heatmap(self.dropdown.currentText(), self.errors_dict, self.tensor1_2d, self.tensor2_2d)  # Call draw_heatmap with the current selected text