import os
import sys
import math
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QMenuBar, QAction, QFileDialog,
                             QToolBar, QToolButton, QFrame, QSizePolicy, QSplitter, QDockWidget, QLineEdit,
                             QPushButton, QComboBox, QMessageBox, QMenu, QScrollArea, QSlider)
from PyQt5.QtCore import Qt, QPoint, QTimer, QSize
from PyQt5.QtGui import QMouseEvent, QPainter, QColor, QPen, QFont, QIcon, QPixmap

import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

class CameraObject:
    """Represents a camera in the 3D scene that can be positioned and oriented"""
    def __init__(self, name="Camera", position=(0, 5, 15), focal_point=(0, 0, 0), view_up=(0, 1, 0)):
        self.name = name
        self.position = position
        self.focal_point = focal_point
        self.view_up = view_up
        
        # Create camera representation (pyramid for perspective view)
        self.actor = self.create_camera_actor()
        self.actor.SetPosition(position)
        
        # Store the actual VTK camera for this camera object
        self.vtk_camera = vtk.vtkCamera()
        self.update_vtk_camera()
        
    def create_camera_actor(self):
        """Create a pyramid-shaped actor to represent the camera in the scene"""
        # Create points for pyramid (camera shape)
        points = vtk.vtkPoints()
        points.InsertNextPoint(0, 0, 0)      # Base center
        points.InsertNextPoint(-1, -1, -2)   # Base corner 1
        points.InsertNextPoint(1, -1, -2)    # Base corner 2
        points.InsertNextPoint(1, 1, -2)     # Base corner 3
        points.InsertNextPoint(-1, 1, -2)    # Base corner 4
        points.InsertNextPoint(0, 0, 2)      # Tip (lens direction)
        
        # Create the pyramid faces
        faces = vtk.vtkCellArray()
        
        # Base (quad)
        base = vtk.vtkPolygon()
        base.GetPointIds().SetNumberOfIds(4)
        base.GetPointIds().SetId(0, 1)
        base.GetPointIds().SetId(1, 2)
        base.GetPointIds().SetId(2, 3)
        base.GetPointIds().SetId(3, 4)
        faces.InsertNextCell(base)
        
        # Side faces (triangles)
        side1 = vtk.vtkTriangle()
        side1.GetPointIds().SetId(0, 0)
        side1.GetPointIds().SetId(1, 1)
        side1.GetPointIds().SetId(2, 2)
        faces.InsertNextCell(side1)
        
        side2 = vtk.vtkTriangle()
        side2.GetPointIds().SetId(0, 0)
        side2.GetPointIds().SetId(1, 2)
        side2.GetPointIds().SetId(2, 3)
        faces.InsertNextCell(side2)
        
        side3 = vtk.vtkTriangle()
        side3.GetPointIds().SetId(0, 0)
        side3.GetPointIds().SetId(1, 3)
        side3.GetPointIds().SetId(2, 4)
        faces.InsertNextCell(side3)
        
        side4 = vtk.vtkTriangle()
        side4.GetPointIds().SetId(0, 0)
        side4.GetPointIds().SetId(1, 4)
        side4.GetPointIds().SetId(2, 1)
        faces.InsertNextCell(side4)
        
        # Lens direction triangle
        lens = vtk.vtkTriangle()
        lens.GetPointIds().SetId(0, 5)
        lens.GetPointIds().SetId(1, 1)
        lens.GetPointIds().SetId(2, 2)
        faces.InsertNextCell(lens)
        
        # Create polydata
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(faces)
        
        # Create mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        
        # Create actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 0.8, 0.2)  # Orange color for camera
        actor.GetProperty().SetOpacity(0.3)
        
        return actor
        
    def update_vtk_camera(self):
        """Update the VTK camera to match this camera object's properties"""
        self.vtk_camera.SetPosition(self.position)
        self.vtk_camera.SetFocalPoint(self.focal_point)
        self.vtk_camera.SetViewUp(self.view_up)
        self.vtk_camera.SetViewAngle(30)  # Field of view
        
    def set_position(self, position):
        """Set camera position and update both actor and VTK camera"""
        self.position = position
        self.actor.SetPosition(position)
        self.update_vtk_camera()
        
    def set_focal_point(self, focal_point):
        """Set camera focal point and update VTK camera"""
        self.focal_point = focal_point
        self.update_vtk_camera()
        
    def set_view_up(self, view_up):
        """Set camera view up vector and update VTK camera"""
        self.view_up = view_up
        self.update_vtk_camera()
        
    def get_view_direction(self):
        """Calculate and return the view direction vector"""
        direction = [
            self.focal_point[0] - self.position[0],
            self.focal_point[1] - self.position[1],
            self.focal_point[2] - self.position[2]
        ]
        length = math.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
        if length > 0:
            return [d/length for d in direction]
        return [0, 0, -1]  # Default: looking down -Z axis

    def get_view_matrix(self):
        """Get the camera's view transformation matrix"""
        return self.vtk_camera.GetViewTransformMatrix()

class ObjectCreationPanel(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Create Objects", parent)
        self.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.setMinimumWidth(250)
        
        # Create content
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(3)
        
        # Store reference to VTK widget
        self.vtk_widget = None
        
        # Create collapsible sections
        self.geometric_section = self.create_collapsible_section("Geometric Objects", [
            'Sphere', 'Cube', 'Pyramid', 'Torus', 'Cylinder', 'Cone'
        ])
        
        self.cell_based_section = self.create_collapsible_section("Cell Based Objects", [
            'Convex Point Set', 'Voxel', 'Hexahedron', 'Polyhedron'
        ])
        
        self.source_format_section = self.create_collapsible_section("Source Formats", [
            'Tetrahedron', 'Octahedron', 'Dodecahedron', 'Icosahedron'
        ])
        
        self.parametric_section = self.create_collapsible_section("Parametric Objects", [
            'Klein Bottle', 'Mobius Strip', 'Super Toroid', 
            'Super Ellipsoid'
        ])
        
        self.isosurface_section = self.create_collapsible_section("Isosurface Objects", [
            'Gyroid', 'Schwarz Primitive', 'Schwarz Diamond', 
            'Schoen IWP', 'Fischer Koch S'
        ])
        
        self.camera_section = self.create_collapsible_section("Cameras", [
            'Camera'
        ])
        
        # Add all sections to layout
        layout.addWidget(self.geometric_section)
        layout.addWidget(self.cell_based_section)
        layout.addWidget(self.source_format_section)
        layout.addWidget(self.parametric_section)
        layout.addWidget(self.isosurface_section)
        layout.addWidget(self.camera_section)
        
        # Add spacer at bottom
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(spacer)
        
        self.setWidget(content)
        
        # Apply styling
        self.setStyleSheet("""
            QDockWidget {
                background-color: #2b2b2b;
                color: white;
                font-size: 9px;
            }
            QDockWidget::title {
                background-color: #323232;
                padding: 3px;
                text-align: center;
            }
        """)
    
    def create_collapsible_section(self, title, object_types):
        """Create a collapsible section with dropdown and create button"""
        # Main section widget
        section_widget = QWidget()
        section_layout = QVBoxLayout(section_widget)
        section_layout.setContentsMargins(0, 0, 0, 0)
        section_layout.setSpacing(2)
        
        # Header (clickable to collapse/expand)
        header = QToolButton()
        header.setText(f"▼ {title}")
        header.setToolButtonStyle(Qt.ToolButtonTextOnly)
        header.setCheckable(True)
        header.setChecked(True)
        header.setStyleSheet("""
            QToolButton {
                background-color: #404040;
                color: white;
                border: 1px solid #505050;
                border-radius: 3px;
                padding: 5px;
                text-align: left;
                font-weight: bold;
            }
            QToolButton:checked {
                background-color: #505050;
            }
            QToolButton:hover {
                background-color: #484848;
            }
        """)
        
        # Content frame (collapsible)
        content_frame = QFrame()
        content_frame.setFrameStyle(QFrame.NoFrame)
        content_frame.setStyleSheet("background-color: #383838; border-radius: 3px; margin: 2px;")
        content_layout = QVBoxLayout(content_frame)
        content_layout.setContentsMargins(8, 8, 8, 8)
        content_layout.setSpacing(4)
        
        # Dropdown for object types
        dropdown = QComboBox()
        dropdown.addItems(object_types)
        dropdown.setStyleSheet("""
            QComboBox {
                background-color: #2b2b2b;
                color: white;
                border: 1px solid #404040;
                border-radius: 3px;
                padding: 3px;
                min-height: 20px;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid white;
                width: 0px;
                height: 0px;
            }
            QComboBox QAbstractItemView {
                background-color: #2b2b2b;
                color: white;
                border: 1px solid #404040;
                selection-background-color: #505050;
            }
        """)
        
        # Create button
        create_btn = QPushButton("Create")
        create_btn.setStyleSheet("""
            QPushButton {
                background-color: #505050;
                color: white;
                border: 1px solid #606060;
                border-radius: 3px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #585858;
            }
            QPushButton:pressed {
                background-color: #404040;
            }
        """)
        
        # Connect button to creation function
        create_btn.clicked.connect(lambda: self.create_object(title, dropdown.currentText()))
        
        # Add widgets to content layout
        content_layout.addWidget(dropdown)
        content_layout.addWidget(create_btn)
        
        # Add to section layout
        section_layout.addWidget(header)
        section_layout.addWidget(content_frame)
        
        # Connect header to toggle content
        header.clicked.connect(lambda checked: self.toggle_section_content(header, content_frame))
        
        return section_widget
    
    def toggle_section_content(self, header, content_frame):
        """Toggle section content visibility"""
        if header.isChecked():
            content_frame.show()
            header.setText(header.text().replace("▶", "▼"))
        else:
            content_frame.hide()
            header.setText(header.text().replace("▼", "▶"))
    
    def set_vtk_widget(self, vtk_widget):
        """Set reference to VTK widget"""
        self.vtk_widget = vtk_widget
    
    def create_object(self, category, object_type):
        """Create the selected object type"""
        if not self.vtk_widget:
            print("No VTK widget reference!")
            return
        
        print(f"Creating {category}: {object_type}")
        
        # Map object types to creation methods
        object_creation_map = {
            'Geometric Objects': {
                'Sphere': 'sphere',
                'Cube': 'cube',
                'Pyramid': 'pyramid', 
                'Torus': 'torus',
                'Cylinder': 'cylinder',
                'Cone': 'cone'
            },
            'Cell Based Objects': {
                'Convex Point Set': 'convex_point',
                'Voxel': 'voxel',
                'Hexahedron': 'hexahedron',
                'Polyhedron': 'polyhedron'
            },
            'Source Formats': {
                'Tetrahedron': 'tetrahedron',
                'Octahedron': 'octahedron',
                'Dodecahedron': 'dodecahedron',
                'Icosahedron': 'icosahedron'
            },
            'Parametric Objects': {
                'Klein Bottle': 'klein',
                'Mobius Strip': 'mobius',
                'Super Toroid': 'super_toroid',
                'Super Ellipsoid': 'super_ellipsoid'
            },
            'Isosurface Objects': {
                'Gyroid': 'gyroid',
                'Schwarz Primitive': 'schwarz_primitive',
                'Schwarz Diamond': 'schwarz_diamond',
                'Schoen IWP': 'schoen_iwp',
                'Fischer Koch S': 'fischer_koch'
            },
            'Cameras': {  # NEW: Camera creation
                'Camera': 'camera'
            }
        }
        
        # Get the object creation key
        creation_key = object_creation_map.get(category, {}).get(object_type)
        if creation_key:
            self.vtk_widget.create_object(creation_key)
        else:
            print(f"Object creation not implemented: {category} - {object_type}")
            
class LightPanel(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Lights", parent)
        self.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.setMinimumWidth(220)

        self.vtk_widget = None

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Header
        header = QLabel("Add Lights")
        header.setStyleSheet("color: #cccccc; font-weight: bold;")
        layout.addWidget(header)

        # Row of icon buttons
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)

        def make_button(icon_func, tooltip, light_type):
            btn = QToolButton()
            btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
            btn.setIcon(icon_func())
            btn.setIconSize(QSize(24, 24))
            btn.setToolTip(tooltip)
            btn.setCheckable(False)
            btn.setStyleSheet("""
                QToolButton {
                    background-color: #404040;
                    border: 1px solid #505050;
                    border-radius: 4px;
                    padding: 3px;
                }
                QToolButton:hover {
                    background-color: #505050;
                }
                QToolButton:pressed {
                    background-color: #303030;
                }
            """)
            btn.clicked.connect(lambda: self.create_light(light_type))
            return btn

        # Icons (Option A: white line icons, minimal)
        row_layout.addWidget(make_button(self.create_point_icon, "Point Light", "point"))
        row_layout.addWidget(make_button(self.create_sun_icon, "Sun (Directional) Light", "sun"))
        row_layout.addWidget(make_button(self.create_spot_icon, "Spot Light", "spot"))
        row_layout.addWidget(make_button(self.create_area_icon, "Area Light", "area"))
        row_layout.addWidget(make_button(self.create_mesh_icon, "Mesh Light", "mesh"))
        row_layout.addWidget(make_button(self.create_world_icon, "World Light", "world"))

        layout.addWidget(row)

        # Spacer so panel can scroll if you add more later
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(spacer)

        self.setWidget(content)

        self.setStyleSheet("""
            QDockWidget {
                background-color: #2b2b2b;
                color: white;
                font-size: 9px;
            }
            QDockWidget::title {
                background-color: #323232;
                padding: 3px;
                text-align: center;
            }
        """)

    def set_vtk_widget(self, vtk_widget):
        self.vtk_widget = vtk_widget

    def create_light(self, light_type: str):
        if not self.vtk_widget:
            print("LightPanel: no VTK widget set.")
            return
        self.vtk_widget.create_light(light_type)

    # ===== Icon drawing helpers (Option A, white line icons) =====
    # All icons are 32x32 transparent pixmaps with white QPen

    def _base_pixmap(self):
        pm = QPixmap(32, 32)
        pm.fill(Qt.transparent)
        painter = QPainter(pm)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        return pm, painter

    def create_point_icon(self):
        pm, p = self._base_pixmap()
        # Small center circle
        p.drawEllipse(14, 14, 4, 4)
        # Short rays
        p.drawLine(16, 8, 16, 4)
        p.drawLine(16, 24, 16, 28)
        p.drawLine(8, 16, 4, 16)
        p.drawLine(24, 16, 28, 16)
        p.end()
        return QIcon(pm)

    def create_sun_icon(self):
        pm, p = self._base_pixmap()
        # Bigger circle
        p.drawEllipse(10, 10, 12, 12)
        # Rays at 8 directions
        rays = [
            (16, 4, 16, 0),
            (16, 28, 16, 32),
            (4, 16, 0, 16),
            (28, 16, 32, 16),
            (7, 7, 3, 3),
            (25, 7, 29, 3),
            (7, 25, 3, 29),
            (25, 25, 29, 29),
        ]
        for x1, y1, x2, y2 in rays:
            p.drawLine(x1, y1, x2, y2)
        p.end()
        return QIcon(pm)

    def create_spot_icon(self):
        pm, p = self._base_pixmap()
        # Cone (lamp)
        p.drawLine(10, 24, 16, 8)
        p.drawLine(22, 24, 16, 8)
        p.drawLine(10, 24, 22, 24)
        # Beam triangle
        p.setPen(QPen(QColor(255, 255, 255, 180), 1))
        p.drawLine(12, 24, 6, 30)
        p.drawLine(20, 24, 26, 30)
        p.drawLine(6, 30, 26, 30)
        p.end()
        return QIcon(pm)

    def create_area_icon(self):
        pm, p = self._base_pixmap()
        # Rectangle panel
        p.drawRect(8, 10, 16, 12)
        # Small arrows outward (showing area emission)
        p.drawLine(8, 16, 2, 16)
        p.drawLine(24, 16, 30, 16)
        p.end()
        return QIcon(pm)

    def create_mesh_icon(self):
        pm, p = self._base_pixmap()
        # Wireframe cube
        p.drawRect(8, 8, 16, 16)
        p.drawRect(10, 10, 12, 12)
        p.drawLine(8, 8, 10, 10)
        p.drawLine(24, 8, 22, 10)
        p.drawLine(8, 24, 10, 22)
        p.drawLine(24, 24, 22, 22)
        p.end()
        return QIcon(pm)

    def create_world_icon(self):
        pm, p = self._base_pixmap()
        # Outer circle
        p.drawEllipse(6, 6, 20, 20)
        # Latitude lines
        p.drawArc(6, 11, 20, 10, 0, 180 * 16)
        p.drawArc(6, 11, 20, 10, 180 * 16, 180 * 16)
        # Longitude lines
        p.drawArc(11, 6, 10, 20, 90 * 16, 180 * 16)
        p.drawArc(11, 6, 10, 20, 270 * 16, 180 * 16)
        p.end()
        return QIcon(pm)

class MeasurementTool:
    def __init__(self, renderer):
        self.renderer = renderer
        self.is_active = False
        self.measurement_mode = None
        self.points = []  # Store measurement points
        self.actors = []  # Store measurement actors (spheres, lines, text)
        self.dragging_point = None
        self.drag_start_pos = None
        self.is_dragging_new_point = False  # NEW: Track if we're dragging a new measurement
        
        # Measurement properties
        self.line_color = (1.0, 1.0, 0.0)  # Yellow
        self.sphere_color = (0.8, 0.8, 1.0)  # Light blue
        self.sphere_radius = 0.3
        self.line_width = 3.0
        
        # Reference plane for universal measurement (default: XY plane at Z=0)
        self.reference_plane = [0, 0, 1, 0]  # [A, B, C, D] for Ax + By + Cz + D = 0
        
    def activate(self):
        """Activate measurement tool"""
        self.is_active = True
        self.measurement_mode = 'distance'
        self.clear_measurements()
        print("Measurement tool activated - Click and drag to measure")
        
    def deactivate(self):
        """Deactivate measurement tool"""
        self.is_active = False
        self.measurement_mode = None
        self.dragging_point = None
        self.is_dragging_new_point = False
        print("Measurement tool deactivated")
        
    def clear_measurements(self):
        """Clear all measurements"""
        self.points.clear()
        for actor in self.actors:
            self.renderer.RemoveActor(actor)
        self.actors.clear()
        self.is_dragging_new_point = False
        
    def set_reference_plane(self, normal, point):
        """Set the reference plane for measurements"""
        # normal: [A, B, C], point: [x, y, z]
        # Plane equation: A(x - x0) + B(y - y0) + C(z - z0) = 0
        # => Ax + By + Cz - (Ax0 + By0 + Cz0) = 0
        # So D = -(Ax0 + By0 + Cz0)
        A, B, C = normal
        x0, y0, z0 = point
        D = -(A*x0 + B*y0 + C*z0)
        self.reference_plane = [A, B, C, D]
        
    def get_world_position_from_mouse(self, mouse_x, mouse_y):
        """Convert mouse coordinates to 3D world position using plane intersection"""
        renderer_size = self.renderer.GetSize()
        if renderer_size[0] == 0 or renderer_size[1] == 0:
            return None
            
        # Convert to VTK display coordinates
        display_x = mouse_x
        display_y = renderer_size[1] - mouse_y  # Flip Y
        
        # Get camera and viewport information
        camera = self.renderer.GetActiveCamera()
        renderer = self.renderer
        
        # Convert display coordinates to world coordinates using plane intersection
        world_point = [0.0, 0.0, 0.0]
        
        # Use vtkWorldPointPicker for more reliable picking
        picker = vtk.vtkWorldPointPicker()
        picker.Pick(display_x, display_y, 0, renderer)
        pick_pos = picker.GetPickPosition()
        
        # If we hit something, use that position
        if picker.GetPickPosition() != [0, 0, 0]:
            return pick_pos
        
        # If no object was hit, use plane intersection method
        # Get camera position and view direction
        camera_pos = camera.GetPosition()
        camera_focus = camera.GetFocalPoint()
        
        # Calculate view direction
        view_dir = [
            camera_focus[0] - camera_pos[0],
            camera_focus[1] - camera_pos[1], 
            camera_focus[2] - camera_pos[2]
        ]
        
        # Normalize view direction
        length = math.sqrt(view_dir[0]**2 + view_dir[1]**2 + view_dir[2]**2)
        if length > 0:
            view_dir = [view_dir[0]/length, view_dir[1]/length, view_dir[2]/length]
        
        # Convert display coordinates to a ray
        renderer.SetDisplayPoint(display_x, display_y, 0)
        renderer.DisplayToWorld()
        world_point1 = renderer.GetWorldPoint()
        world_point1 = [world_point1[0]/world_point1[3], world_point1[1]/world_point1[3], world_point1[2]/world_point1[3]]
        
        renderer.SetDisplayPoint(display_x, display_y, 1)
        renderer.DisplayToWorld()
        world_point2 = renderer.GetWorldPoint()
        world_point2 = [world_point2[0]/world_point2[3], world_point2[1]/world_point2[3], world_point2[2]/world_point2[3]]
        
        # Calculate ray direction
        ray_dir = [
            world_point2[0] - world_point1[0],
            world_point2[1] - world_point1[1],
            world_point2[2] - world_point1[2]
        ]
        
        # Normalize ray direction
        length = math.sqrt(ray_dir[0]**2 + ray_dir[1]**2 + ray_dir[2]**2)
        if length > 0:
            ray_dir = [ray_dir[0]/length, ray_dir[1]/length, ray_dir[2]/length]
        
        # Intersect ray with reference plane (default: Z=0 plane)
        # For now, use a plane parallel to camera view plane at a reasonable distance
        # This creates a "virtual ground plane" effect
        
        # Use a plane that's perpendicular to the camera's view direction
        # and positioned at the current focal point distance
        plane_normal = view_dir
        plane_point = camera_focus
        
        # Calculate intersection between ray and plane
        # Plane equation: (P - P0) · N = 0
        # Ray equation: P = O + tD
        # Substitute: (O + tD - P0) · N = 0
        # => t = ((P0 - O) · N) / (D · N)
        
        O = world_point1  # Ray origin
        D = ray_dir       # Ray direction
        P0 = plane_point  # Point on plane
        N = plane_normal  # Plane normal
        
        dot_DN = D[0]*N[0] + D[1]*N[1] + D[2]*N[2]
        
        if abs(dot_DN) > 1e-6:  # Avoid division by zero
            t = ((P0[0] - O[0])*N[0] + (P0[1] - O[1])*N[1] + (P0[2] - O[2])*N[2]) / dot_DN
            if t >= 0:  # Intersection in front of camera
                intersection_point = [
                    O[0] + t * D[0],
                    O[1] + t * D[1], 
                    O[2] + t * D[2]
                ]
                return intersection_point
        
        # Fallback: use a point along the view direction
        return [
            camera_pos[0] + view_dir[0] * 10,
            camera_pos[1] + view_dir[1] * 10,
            camera_pos[2] + view_dir[2] * 10
        ]
        
    def handle_click(self, mouse_x, mouse_y):
        """Handle mouse click in measurement mode - FIXED VERSION"""
        if not self.is_active:
            return False
            
        world_pos = self.get_world_position_from_mouse(mouse_x, mouse_y)
        if not world_pos:
            return False
            
        print(f"Measurement click at screen: ({mouse_x}, {mouse_y}), world: {world_pos}")
        
        # Check if clicking on existing point
        point_idx = self.get_point_at_position(world_pos)
        if point_idx is not None:
            # Start dragging existing point
            self.dragging_point = point_idx
            self.drag_start_pos = world_pos
            print(f"Started dragging existing point {point_idx}")
            return True
        else:
            # Start creating new measurement
            if len(self.points) == 0:
                # First point - place it immediately
                self.points.append(world_pos)
                self.create_sphere(world_pos)
                print("Placed first measurement point")
                
                # Start dragging to create second point
                self.is_dragging_new_point = True
                self.dragging_point = 1  # We'll create point at index 1
                self.drag_start_pos = world_pos
                return True
            elif len(self.points) == 1:
                # Second point - place it immediately
                self.points.append(world_pos)
                self.create_sphere(world_pos)
                self.create_line(self.points[0], self.points[1])
                self.create_distance_text(self.points[0], self.points[1])
                print("Placed second measurement point")
                return True
            else:
                # Already have 2 points - start new measurement
                self.clear_measurements()
                self.points.append(world_pos)
                self.create_sphere(world_pos)
                self.is_dragging_new_point = True
                self.dragging_point = 1
                self.drag_start_pos = world_pos
                print("Started new measurement")
                return True
                    
        return False
        
    def handle_drag(self, mouse_x, mouse_y):
        """Handle mouse drag in measurement mode - FIXED VERSION"""
        if not self.is_active:
            return False
        
        world_pos = self.get_world_position_from_mouse(mouse_x, mouse_y)
        if not world_pos:
            return False
            
        if self.is_dragging_new_point:
            # We're dragging to create a new point
            if len(self.points) == 1:
                # Create temporary visualization for the dragging
                self.update_temporary_measurement(world_pos)
                return True
                
        elif self.dragging_point is not None:
            # Dragging existing point
            self.points[self.dragging_point] = world_pos
            self.update_visualization()
            return True
            
        return False
        
    def handle_release(self, mouse_x, mouse_y):
        """Handle mouse release in measurement mode - FIXED VERSION"""
        if not self.is_active:
            return False
        
        world_pos = self.get_world_position_from_mouse(mouse_x, mouse_y)
        if not world_pos:
            return False
            
        if self.is_dragging_new_point and len(self.points) == 1:
            # Finish creating the second point
            self.points.append(world_pos)
            self.create_sphere(world_pos)
            self.create_line(self.points[0], self.points[1])
            self.create_distance_text(self.points[0], self.points[1])
            print("Completed measurement with drag")
            
        self.is_dragging_new_point = False
        self.dragging_point = None
        self.drag_start_pos = None
        return True
        
    def update_temporary_measurement(self, world_pos):
        """Update temporary visualization during drag"""
        # Clear any existing temporary visualization
        actors_to_remove = []
        for actor in self.actors:
            if hasattr(actor, '_is_temporary') and actor._is_temporary:
                actors_to_remove.append(actor)
                
        for actor in actors_to_remove:
            self.renderer.RemoveActor(actor)
            self.actors.remove(actor)
            
        # Create temporary line
        if len(self.points) == 1:
            temp_line = self.create_line(self.points[0], world_pos)
            temp_line._is_temporary = True
            
            # Create temporary distance text
            distance = math.sqrt(
                (world_pos[0] - self.points[0][0])**2 +
                (world_pos[1] - self.points[0][1])**2 +
                (world_pos[2] - self.points[0][2])**2
            )
            
            mid_point = [
                (self.points[0][0] + world_pos[0]) / 2,
                (self.points[0][1] + world_pos[1]) / 2,
                (self.points[0][2] + world_pos[2]) / 2
            ]
            
            temp_text = self.create_distance_text_at_position(mid_point, distance)
            temp_text._is_temporary = True
        
    def create_distance_text_at_position(self, position, distance):
        """Create distance text at specific position"""
        text_source = vtk.vtkVectorText()
        text_source.SetText(f"{distance:.2f}")
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(text_source.GetOutputPort())
        
        follower = vtk.vtkFollower()
        follower.SetMapper(mapper)
        follower.SetPosition(position)
        follower.GetProperty().SetColor(1.0, 1.0, 0.0)
        follower.SetScale(0.3, 0.3, 0.3)
        follower.SetCamera(self.renderer.GetActiveCamera())
        
        self.renderer.AddActor(follower)
        self.actors.append(follower)
        return follower

    def get_point_at_position(self, world_pos, tolerance=0.5):
        """Check if a point exists near the given world position"""
        for i, point in enumerate(self.points):
            distance = math.sqrt(
                (point[0] - world_pos[0])**2 +
                (point[1] - world_pos[1])**2 +
                (point[2] - world_pos[2])**2
            )
            if distance < tolerance:
                return i
        return None
        
    def create_sphere(self, position):
        """Create a transparent sphere at the given position"""
        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetCenter(position)
        sphere_source.SetRadius(self.sphere_radius)
        sphere_source.SetPhiResolution(16)
        sphere_source.SetThetaResolution(16)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere_source.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(self.sphere_color)
        actor.GetProperty().SetOpacity(0.3)  # Transparent
        
        self.renderer.AddActor(actor)
        self.actors.append(actor)
        return actor
        
    def create_line(self, point1, point2):
        """Create a line between two points"""
        line_source = vtk.vtkLineSource()
        line_source.SetPoint1(point1)
        line_source.SetPoint2(point2)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(line_source.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(self.line_color)
        actor.GetProperty().SetLineWidth(self.line_width)
        
        self.renderer.AddActor(actor)
        self.actors.append(actor)
        return actor
        
    def create_distance_text(self, point1, point2):
        """Create text showing distance between points"""
        distance = math.sqrt(
            (point2[0] - point1[0])**2 +
            (point2[1] - point1[1])**2 +
            (point2[2] - point1[2])**2
        )
        
        # Calculate midpoint for text position
        mid_point = [
            (point1[0] + point2[0]) / 2,
            (point1[1] + point2[1]) / 2,
            (point1[2] + point2[2]) / 2
        ]
        
        return self.create_distance_text_at_position(mid_point, distance)
        
    def update_visualization(self):
        """Update all measurement visualization"""
        # Clear existing visualization (except spheres)
        actors_to_remove = []
        for actor in self.actors:
            if not isinstance(actor.GetMapper().GetInputConnection(0, 0).GetProducer(), vtk.vtkSphereSource):
                actors_to_remove.append(actor)
                
        for actor in actors_to_remove:
            self.renderer.RemoveActor(actor)
            self.actors.remove(actor)
            
        # Recreate lines and text
        if len(self.points) >= 2:
            self.create_line(self.points[0], self.points[1])
            self.create_distance_text(self.points[0], self.points[1])

class TransformWidget(QWidget):
    def __init__(self, vtk_widget=None):
        super().__init__()
        self.vtk_widget = vtk_widget
        self.selected_actor = None
        self.updates_paused = False  # Add this flag
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)
        
        # Transform header with normal font
        self.transform_header = QToolButton()
        self.transform_header.setText("Transform ▼")
        self.transform_header.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.transform_header.setCheckable(True)
        self.transform_header.setChecked(True)
        
        self.transform_header.setStyleSheet("""
            QToolButton {
                background-color: #404040;
                color: white;
                border: 1px solid #505050;
                border-radius: 3px;
                padding: 5px;
                text-align: left;
            }
            QToolButton:checked {
                background-color: #505050;
            }
            QToolButton:hover {
                background-color: #484848;
            }
        """)
        self.transform_header.clicked.connect(self.toggle_transform_content)
        layout.addWidget(self.transform_header)
        
        # Transform content frame
        self.transform_content = QFrame()
        self.transform_content.setFrameStyle(QFrame.NoFrame)
        self.transform_content.setStyleSheet("background-color: #383838; border-radius: 3px; margin: 2px;")
        content_layout = QVBoxLayout(self.transform_content)
        content_layout.setContentsMargins(8, 8, 8, 8)
        content_layout.setSpacing(4)
        
        # Location section with normal font
        location_label = QLabel("Location")
        location_label.setStyleSheet("color: #cccccc; font-weight: bold; margin-top: 5px;")
        content_layout.addWidget(location_label)
        
        # X, Y, Z input fields for Location
        self.location_x = self.create_vector_input("X:")
        self.location_y = self.create_vector_input("Y:") 
        self.location_z = self.create_vector_input("Z:")
        
        content_layout.addWidget(self.location_x)
        content_layout.addWidget(self.location_y)
        content_layout.addWidget(self.location_z)
        
        # Add some spacing between sections
        spacer1 = QWidget()
        spacer1.setFixedHeight(5)
        content_layout.addWidget(spacer1)
        
        # Rotation section with normal font
        rotation_label = QLabel("Rotation")
        rotation_label.setStyleSheet("color: #cccccc; font-weight: bold; margin-top: 5px;")
        content_layout.addWidget(rotation_label)
        
        # X, Y, Z input fields for Rotation
        self.rotation_x = self.create_vector_input("X:")
        self.rotation_y = self.create_vector_input("Y:") 
        self.rotation_z = self.create_vector_input("Z:")
        
        content_layout.addWidget(self.rotation_x)
        content_layout.addWidget(self.rotation_y)
        content_layout.addWidget(self.rotation_z)
        
        # Scale section with normal font
        scale_label = QLabel("Scale")
        scale_label.setStyleSheet("color: #cccccc; font-weight: bold; margin-top: 5px;")
        content_layout.addWidget(scale_label)
        
        # X, Y, Z input fields for Scale
        self.scale_x = self.create_vector_input("X:")
        self.scale_y = self.create_vector_input("Y:") 
        self.scale_z = self.create_vector_input("Z:")
        
        content_layout.addWidget(self.scale_x)
        content_layout.addWidget(self.scale_y)
        content_layout.addWidget(self.scale_z)
        
        # Add some spacing before apply button
        spacer2 = QWidget()
        spacer2.setFixedHeight(5)
        content_layout.addWidget(spacer2)
        
        # Apply button with normal font
        self.apply_btn = QPushButton("Apply Transform")
        self.apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #505050;
                color: white;
                border: 1px solid #606060;
                border-radius: 3px;
                padding: 6px;
                margin-top: 5px;
            }
            QPushButton:hover {
                background-color: #585858;
            }
            QPushButton:pressed {
                background-color: #404040;
            }
            QPushButton:disabled {
                background-color: #383838;
                color: #666666;
            }
        """)
        self.apply_btn.clicked.connect(self.apply_transform)
        content_layout.addWidget(self.apply_btn)
        
        layout.addWidget(self.transform_content)
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_from_selection)
        self.update_timer.start(100)  # Update every 100ms like other functions
        
    def create_vector_input(self, label):
        """Create a labeled vector input field with NORMAL FONT"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        label_widget = QLabel(label)
        label_widget.setFixedWidth(20)
        label_widget.setStyleSheet("color: #aaaaaa;")
        layout.addWidget(label_widget)
        
        input_field = QLineEdit()
        input_field.setPlaceholderText("0.0")
        input_field.setFixedWidth(80)
        input_field.setAlignment(Qt.AlignRight)
        
        input_field.setStyleSheet("""
            QLineEdit {
                background-color: #2b2b2b;
                color: white;
                border: 1px solid #404040;
                border-radius: 3px;
                padding: 3px;
            }
            QLineEdit:focus {
                border: 1px solid #ff8000;
            }
            QLineEdit:disabled {
                background-color: #252525;
                color: #666666;
            }
        """)
        input_field.textChanged.connect(self.on_vector_changed)
        layout.addWidget(input_field)
        
        return widget

    def pause_updates(self):
        """Pause automatic updates from selection (used during scaling)"""
        self.updates_paused = True
        
    def resume_updates(self):
        """Resume automatic updates from selection"""
        self.updates_paused = False

    def toggle_transform_content(self):
        """Toggle transform section visibility"""
        if self.transform_header.isChecked():
            self.transform_content.show()
            self.transform_header.setText("Transform ▼")
        else:
            self.transform_content.hide()
            self.transform_header.setText("Transform ▶")
    
    def update_from_selection(self):
        """Update input fields from selected object - FIXED VERSION"""
        if self.updates_paused:
            return
            
        if not self.vtk_widget or not self.vtk_widget.object_manager.selected_actors:
            self.selected_actor = None
            self.set_inputs_enabled(False)
            return
        
        selected_actor = self.vtk_widget.object_manager.selected_actors[0]
        if selected_actor != self.selected_actor:
            self.selected_actor = selected_actor
            self.set_inputs_enabled(True)
            
        # Update location fields
        position = selected_actor.GetPosition()
        
        # Block signals to prevent recursive updates
        self.location_x.findChild(QLineEdit).blockSignals(True)
        self.location_y.findChild(QLineEdit).blockSignals(True)
        self.location_z.findChild(QLineEdit).blockSignals(True)
        
        self.location_x.findChild(QLineEdit).setText(f"{position[0]:.3f}")
        self.location_y.findChild(QLineEdit).setText(f"{position[1]:.3f}")
        self.location_z.findChild(QLineEdit).setText(f"{position[2]:.3f}")
        
        # Unblock signals
        self.location_x.findChild(QLineEdit).blockSignals(False)
        self.location_y.findChild(QLineEdit).blockSignals(False)
        self.location_z.findChild(QLineEdit).blockSignals(False)
        
        # Update rotation fields
        orientation = selected_actor.GetOrientation()
        
        # Block signals for rotation fields too
        self.rotation_x.findChild(QLineEdit).blockSignals(True)
        self.rotation_y.findChild(QLineEdit).blockSignals(True)
        self.rotation_z.findChild(QLineEdit).blockSignals(True)
        
        self.rotation_x.findChild(QLineEdit).setText(f"{orientation[0]:.3f}")
        self.rotation_y.findChild(QLineEdit).setText(f"{orientation[1]:.3f}")
        self.rotation_z.findChild(QLineEdit).setText(f"{orientation[2]:.3f}")
        
        # Unblock signals
        self.rotation_x.findChild(QLineEdit).blockSignals(False)
        self.rotation_y.findChild(QLineEdit).blockSignals(False)
        self.rotation_z.findChild(QLineEdit).blockSignals(False)
        
        # Update scale fields
        scale = selected_actor.GetScale()
        
        # Block signals for scale fields
        self.scale_x.findChild(QLineEdit).blockSignals(True)
        self.scale_y.findChild(QLineEdit).blockSignals(True)
        self.scale_z.findChild(QLineEdit).blockSignals(True)
        
        self.scale_x.findChild(QLineEdit).setText(f"{scale[0]:.3f}")
        self.scale_y.findChild(QLineEdit).setText(f"{scale[1]:.3f}")
        self.scale_z.findChild(QLineEdit).setText(f"{scale[2]:.3f}")
        
        # Unblock signals
        self.scale_x.findChild(QLineEdit).blockSignals(False)
        self.scale_y.findChild(QLineEdit).blockSignals(False)
        self.scale_z.findChild(QLineEdit).blockSignals(False)
    
    def set_inputs_enabled(self, enabled):
        """Enable or disable input fields"""
        self.location_x.setEnabled(enabled)
        self.location_y.setEnabled(enabled)
        self.location_z.setEnabled(enabled)
        self.rotation_x.setEnabled(enabled)
        self.rotation_y.setEnabled(enabled)
        self.rotation_z.setEnabled(enabled)
        self.scale_x.setEnabled(enabled)
        self.scale_y.setEnabled(enabled)
        self.scale_z.setEnabled(enabled)
        self.apply_btn.setEnabled(enabled)
        
        if not enabled:
            self.location_x.findChild(QLineEdit).setText("")
            self.location_y.findChild(QLineEdit).setText("")
            self.location_z.findChild(QLineEdit).setText("")
            self.rotation_x.findChild(QLineEdit).setText("")
            self.rotation_y.findChild(QLineEdit).setText("")
            self.rotation_z.findChild(QLineEdit).setText("")
            self.scale_x.findChild(QLineEdit).setText("")
            self.scale_y.findChild(QLineEdit).setText("")
            self.scale_z.findChild(QLineEdit).setText("")
    
    def on_vector_changed(self):
        """Handle manual input changes for location, rotation and scale"""
        if not self.selected_actor:
            return
        
        try:
            # Update location
            loc_x = float(self.location_x.findChild(QLineEdit).text() or "0")
            loc_y = float(self.location_y.findChild(QLineEdit).text() or "0")
            loc_z = float(self.location_z.findChild(QLineEdit).text() or "0")
            
            # Update rotation
            rot_x = float(self.rotation_x.findChild(QLineEdit).text() or "0")
            rot_y = float(self.rotation_y.findChild(QLineEdit).text() or "0")
            rot_z = float(self.rotation_z.findChild(QLineEdit).text() or "0")
            
            # Update scale
            scale_x = float(self.scale_x.findChild(QLineEdit).text() or "1")
            scale_y = float(self.scale_y.findChild(QLineEdit).text() or "1")
            scale_z = float(self.scale_z.findChild(QLineEdit).text() or "1")
            
            # Update object position, orientation and scale
            self.selected_actor.SetPosition(loc_x, loc_y, loc_z)
            self.selected_actor.SetOrientation(rot_x, rot_y, rot_z)
            self.selected_actor.SetScale(scale_x, scale_y, scale_z)
            
            # Update outline
            if self.selected_actor in self.vtk_widget.object_manager.outline_actors:
                outline_actor = self.vtk_widget.object_manager.outline_actors[self.selected_actor]
                outline_actor.SetPosition(loc_x, loc_y, loc_z)
                outline_actor.SetOrientation(rot_x, rot_y, rot_z)
                outline_actor.SetScale(scale_x, scale_y, scale_z)
            
            # Update gizmo positions
            if self.vtk_widget.object_manager.move_gizmo.is_visible:
                self.vtk_widget.object_manager.move_gizmo.update_position()
            if self.vtk_widget.object_manager.rotate_gizmo.is_visible:
                self.vtk_widget.object_manager.rotate_gizmo.update_position()
            if self.vtk_widget.object_manager.scale_gizmo.is_visible:
                self.vtk_widget.object_manager.scale_gizmo.update_position()
                
            if hasattr(self.vtk_widget, "light_manager"):
                self.vtk_widget.light_manager.sync_light_for_actor(self.selected_actor)
                                
            # Force render
            self.vtk_widget.render_window.Render()
            
        except ValueError:
            pass  # Ignore invalid input
    
    def apply_transform(self):
        """Apply the current transform values"""
        self.on_vector_changed()
        
class LightingWidget(QWidget):
    """Small UI panel that controls LightManager modes."""
    def __init__(self, vtk_widget=None):
        super().__init__()
        self.vtk_widget = vtk_widget
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Header
        self.header = QToolButton()
        self.header.setText("Lighting ▼")
        self.header.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.header.setCheckable(True)
        self.header.setChecked(True)
        self.header.setStyleSheet("""
            QToolButton {
                background-color: #404040;
                color: white;
                border: 1px solid #505050;
                border-radius: 3px;
                padding: 5px;
                text-align: left;
                margin-top: 5px;
            }
            QToolButton:checked {
                background-color: #505050;
            }
            QToolButton:hover {
                background-color: #484848;
            }
        """)
        self.header.clicked.connect(self.toggle_content)
        layout.addWidget(self.header)

        # Content frame
        self.content = QFrame()
        self.content.setFrameStyle(QFrame.NoFrame)
        self.content.setStyleSheet("background-color: #383838; border-radius: 3px; margin: 2px;")
        c_layout = QVBoxLayout(self.content)
        c_layout.setContentsMargins(8, 8, 8, 8)
        c_layout.setSpacing(4)
        
        # Label for selected light
        self.selected_light_label = QLabel("Selected Light: None")
        self.selected_light_label.setStyleSheet("color: #cccccc; font-weight: bold;")
        c_layout.addWidget(self.selected_light_label)

        # Intensity slider
        int_row = QWidget()
        int_layout = QHBoxLayout(int_row)
        int_layout.setContentsMargins(0, 0, 0, 0)
        int_layout.setSpacing(4)

        int_label = QLabel("Intensity")
        int_label.setStyleSheet("color: #cccccc;")
        int_layout.addWidget(int_label)

        self.intensity_slider = QSlider(Qt.Horizontal)
        self.intensity_slider.setMinimum(0)     # 0.0
        self.intensity_slider.setMaximum(200)   # 2.0
        self.intensity_slider.setValue(100)     # default 1.0
        self.intensity_slider.setSingleStep(5)
        self.intensity_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px;
                background: #555555;
            }
            QSlider::handle:horizontal {
                width: 10px;
                background: #ff8000;
                margin: -6px 0;
            }
        """)
        self.intensity_slider.valueChanged.connect(self.on_intensity_changed)
        int_layout.addWidget(self.intensity_slider)

        c_layout.addWidget(int_row)

        # Info label
        self.info_label = QLabel(
            "• Mesh Light: use selected object as emitter.\n"
            "• World Light: brightens entire scene.\n"
            "• Others behave similar to Blender lights."
        )
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color: #aaaaaa; font-size: 9px;")
        c_layout.addWidget(self.info_label)

        layout.addWidget(self.content)

    def toggle_content(self):
        if self.header.isChecked():
            self.content.show()
            self.header.setText("Lighting ▼")
        else:
            self.content.hide()
            self.header.setText("Lighting ▶")

    def set_vtk_widget(self, vtk_widget):
        self.vtk_widget = vtk_widget

    def apply_light(self):
        if not self.vtk_widget or not hasattr(self.vtk_widget, "light_manager"):
            return

        text = self.type_combo.currentText()
        mapping = {
            "Point Light": "point",
            "Sun Light": "sun",
            "Spot Light": "spot",
            "Area Light": "area",
            "Mesh Light": "mesh",
            "World Light": "world"
        }
        mode = mapping.get(text)
        if mode:
            self.vtk_widget.light_manager.set_mode(mode)

    def reset_lights(self):
        if self.vtk_widget and hasattr(self.vtk_widget, "light_manager"):
            self.vtk_widget.light_manager.set_mode("world")
            
    def set_vtk_widget(self, vtk_widget):
        """Called from RightPanel to give us access to VTK + LightManager."""
        self.vtk_widget = vtk_widget

    def update_from_selection(self):
        """Call this periodically or on selection change."""
        if not self.vtk_widget or not self.vtk_widget.object_manager.selected_actors:
            self.selected_light_label.setText("Selected Light: None")
            self.intensity_slider.setEnabled(False)
            return
        
        actor = self.vtk_widget.object_manager.selected_actors[0]
        
        # Check if this actor is a light icon
        lm = getattr(self.vtk_widget, "light_manager", None)
        if lm and actor in lm.light_objects:
            info = lm.light_objects[actor]
            light_type = info["type"].capitalize()
            self.selected_light_label.setText(f"Selected Light: {light_type}")
            self.intensity_slider.setEnabled(True)
        else:
            self.selected_light_label.setText("Selected Light: (not a light)")
            self.intensity_slider.setEnabled(False)

    def on_intensity_changed(self, value):
        """Map slider 0–200 to intensity 0.0–2.0 and send to LightManager."""
        if not self.vtk_widget or not self.vtk_widget.object_manager.selected_actors:
            return
        
        actor = self.vtk_widget.object_manager.selected_actors[0]
        lm = getattr(self.vtk_widget, "light_manager", None)
        if not lm or actor not in lm.light_objects:
            return
        
        intensity = value / 100.0  # 0–200 → 0.0–2.0
        lm.set_intensity_for_actor(actor, intensity)
        
class ViewportGizmo2D(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(120, 120)
        
        # Make it a top-level widget that stays on top
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setStyleSheet("background: transparent; border: none;")
        
        # Store camera angles for orientation
        self.camera_phi = math.radians(45.0)
        self.camera_theta = math.radians(45.0)
        
        # Store reference to main window for reset functionality
        self.main_window = None
        
    def set_main_window(self, main_window):
        """Set reference to main window for reset functionality and positioning"""
        self.main_window = main_window
        self.update_position()
        
    def update_orientation(self, phi, theta):
        """Update the gizmo based on camera orientation"""
        self.camera_phi = phi
        self.camera_theta = theta
        self.update()
        
    def update_position(self):
        """Update gizmo position to stay in top-right corner, above everything"""
        if self.main_window:
            # Get the main window geometry
            main_rect = self.main_window.geometry()
            
            # Calculate position in top-right corner, accounting for the right panel
            right_panel_width = 0
            if hasattr(self.main_window, 'right_panel') and self.main_window.right_panel.isVisible():
                right_panel_width = self.main_window.right_panel.width()
            
            # Position in top-right, above the right panel
            x = main_rect.width() - self.width() - right_panel_width - 10  # 10px margin from right panel
            y = 50  # 50px from top (below menu bar)
            
            self.move(x, y)
        
    def mousePressEvent(self, event):
        """Handle mouse click to reset camera to original position"""
        if event.button() == Qt.LeftButton:
            self.reset_camera_view()
            event.accept()
        else:
            super().mousePressEvent(event)
            
    def reset_camera_view(self):
        """Reset camera to original position (like Blender's Home key)"""
        if self.main_window and hasattr(self.main_window, 'vtk_widget'):
            self.main_window.vtk_widget.reset_view()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # COMPLETELY transparent background - no fill at all
        painter.setCompositionMode(QPainter.CompositionMode_Clear)
        painter.fillRect(self.rect(), Qt.transparent)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        
        center_x = 60
        center_y = 60
        radius = 40
        
        # Draw background circle (semi-transparent gray only)
        painter.setBrush(QColor(80, 80, 80, 150))
        painter.setPen(QPen(QColor(120, 120, 120, 180), 2))
        painter.drawEllipse(center_x - radius, center_y - radius, radius * 2, radius * 2)
        
        # Get the direction vectors for each axis in camera space
        x_dir = self.get_axis_direction(1, 0, 0)   # Right (Green)
        y_dir = self.get_axis_direction(0, 1, 0)   # Forward (Red)  
        z_dir = self.get_axis_direction(0, 0, 1)   # Up (Blue)
        
        # Draw axes with colors
        self.draw_axis(painter, center_x, center_y, 
                      center_x + x_dir[0] * radius, center_y + x_dir[1] * radius, 
                      QColor(80, 255, 80))  # Green - X (Right)
        
        self.draw_axis(painter, center_x, center_y,
                      center_x + y_dir[0] * radius, center_y + y_dir[1] * radius,
                      QColor(255, 80, 80))  # Red - Y (Forward)
        
        self.draw_axis(painter, center_x, center_y,
                      center_x + z_dir[0] * radius, center_y + z_dir[1] * radius,
                      QColor(80, 150, 255))  # Blue - Z (Up)
        
        painter.end()
    
    def get_axis_direction(self, x, y, z):
        """Get the 2D screen direction of a world axis based on camera orientation"""
        sin_phi = math.sin(self.camera_phi)
        cos_phi = math.cos(self.camera_phi)
        sin_theta = math.sin(self.camera_theta)
        cos_theta = math.cos(self.camera_theta)
        
        # Transform world axis to camera view coordinates
        view_x = x * cos_phi - y * sin_phi
        view_y = x * sin_phi * cos_theta + y * cos_phi * cos_theta - z * sin_theta
        
        return (view_x, view_y)
    
    def draw_axis(self, painter, start_x, start_y, end_x, end_y, color):
        """Draw an axis line with arrow head at the END (positive direction)"""
        pen = QPen(color)
        pen.setWidth(4)
        painter.setPen(pen)
        
        # Draw line from center to the positive direction
        painter.drawLine(int(start_x), int(start_y), int(end_x), int(end_y))
        
        # Draw arrow head at the END (positive direction)
        arrow_size = 8
        dx = end_x - start_x
        dy = end_y - start_y
        length = math.sqrt(dx*dx + dy*dy)
        
        if length > 0:
            dx, dy = dx/length, dy/length
            
            # Arrow points at the end
            arrow_x1 = end_x - dx * arrow_size + dy * arrow_size/3
            arrow_y1 = end_y - dy * arrow_size - dx * arrow_size/3
            arrow_x2 = end_x - dx * arrow_size - dy * arrow_size/3
            arrow_y2 = end_y - dy * arrow_size + dx * arrow_size/3
            
            painter.drawLine(int(end_x), int(end_y), int(arrow_x1), int(arrow_y1))
            painter.drawLine(int(end_x), int(end_y), int(arrow_x2), int(arrow_y2))
            
class MoveGizmo:
    def __init__(self, renderer):
        self.renderer = renderer
        self.actors = {}
        self.is_visible = False
        self.selected_actor = None
        self.gizmo_size = 5.0  # Increased size for better visibility
        
        # Store original actor properties for opacity restoration
        self.original_opacities = {}
        
        # Create the gizmo axes
        self.create_gizmo()
    
    def create_gizmo(self):
        """Create the move gizmo with X, Y, Z arrows - LARGER AND MORE VISIBLE"""
        # X-axis (Red) - pointing right
        x_arrow = self.create_arrow([0, 0, 0], [self.gizmo_size, 0, 0], [1.0, 0.0, 0.0])
        # Y-axis (Green) - pointing forward
        y_arrow = self.create_arrow([0, 0, 0], [0, self.gizmo_size, 0], [0.0, 1.0, 0.0])
        # Z-axis (Blue) - pointing up
        z_arrow = self.create_arrow([0, 0, 0], [0, 0, self.gizmo_size], [0.0, 0.4, 1.0])
        
        self.actors['x'] = x_arrow
        self.actors['y'] = y_arrow
        self.actors['z'] = z_arrow
        
    
    def create_arrow(self, start, end, color):
        """Create an arrow actor with proper sizing"""
        # Create arrow source with larger size
        arrow_source = vtk.vtkArrowSource()
        arrow_source.SetTipLength(0.3)  # Larger tip
        arrow_source.SetTipRadius(0.1)   # Larger tip radius
        arrow_source.SetShaftRadius(0.04) # Thicker shaft
        
        # Calculate direction and length
        direction = [end[0] - start[0], end[1] - start[1], end[2] - start[2]]
        length = math.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
        
        if length > 0:
            direction = [d/length for d in direction]
        
        # Create transform to position and orient the arrow
        transform = vtk.vtkTransform()
        transform.Translate(start)
        
        # Calculate rotation - default arrow points along X-axis
        x_axis = [1, 0, 0]  # Default arrow direction
        if direction != x_axis:
            # Calculate rotation axis using cross product
            rotation_axis = [
                x_axis[1] * direction[2] - x_axis[2] * direction[1],
                x_axis[2] * direction[0] - x_axis[0] * direction[2],
                x_axis[0] * direction[1] - x_axis[1] * direction[0]
            ]
            
            # Calculate rotation angle
            dot_product = sum(x_axis[i] * direction[i] for i in range(3))
            rotation_angle = math.acos(max(-1, min(1, dot_product)))
            rotation_angle_deg = math.degrees(rotation_angle)
            
            # Apply rotation
            if rotation_angle_deg > 0.1:
                transform.RotateWXYZ(rotation_angle_deg, rotation_axis)
        
        transform.Scale(length, length, length)
        
        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetTransform(transform)
        transform_filter.SetInputConnection(arrow_source.GetOutputPort())
        
        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(transform_filter.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetLineWidth(3)  # Thicker lines
        
        return actor
    
    def show(self, target_actor):
        """Show gizmo on target actor and reduce object opacity"""
        print(f"MoveGizmo.show() called with actor: {target_actor}")
        self.selected_actor = target_actor
        if target_actor:
            # Store original opacity and reduce it
            original_opacity = target_actor.GetProperty().GetOpacity()
            self.original_opacities[target_actor] = original_opacity
            target_actor.GetProperty().SetOpacity(0.3)  # Reduced opacity for visibility
            print(f"Reduced opacity from {original_opacity} to 0.3")
            
            # Get target position
            position = target_actor.GetPosition()
            print(f"Positioning gizmo at: {position}")
            
            # Position gizmo at target
            for axis_name, axis_actor in self.actors.items():
                axis_actor.SetPosition(position)
                self.renderer.AddActor(axis_actor)
                print(f"Added {axis_name} axis actor to renderer")
            
            self.is_visible = True
            print("Move gizmo is now visible")
            
            # Force render only if render window exists
            self.safe_render()
        else:
            print("MoveGizmo.show(): No target actor provided!")
    
    def hide(self):
        """Hide gizmo and restore object opacity"""
        print("MoveGizmo.hide() called")
        
        # Restore original opacities
        for actor, original_opacity in self.original_opacities.items():
            if actor:  # Check if actor still exists
                actor.GetProperty().SetOpacity(original_opacity)
                print(f"Restored opacity to {original_opacity}")
        self.original_opacities.clear()
        
        # Remove gizmo actors
        for axis_name, axis_actor in self.actors.items():
            self.renderer.RemoveActor(axis_actor)
            print(f"Removed {axis_name} axis actor from renderer")
        
        self.is_visible = False
        self.selected_actor = None
        
        # Force render only if render window exists
        self.safe_render()
    
    def safe_render(self):
        """Safely render only if render window exists"""
        try:
            render_window = self.renderer.GetRenderWindow()
            if render_window:
                render_window.Render()
        except Exception as e:
            print(f"Safe render failed: {e}")
    
    def update_position(self):
        """Update gizmo position to follow selected object"""
        if self.is_visible and self.selected_actor:
            position = self.selected_actor.GetPosition()
            for axis_actor in self.actors.values():
                axis_actor.SetPosition(position)
    
    def get_axis_at_position(self, x, y):
        """Check if gizmo axis is clicked at screen position"""
        if not self.is_visible:
            return None
        
        picker = vtk.vtkPropPicker()
        picker.Pick(x, y, 0, self.renderer)
        
        picked_actor = picker.GetActor()
        if picked_actor in self.actors.values():
            # Find which axis was picked
            for axis, actor in self.actors.items():
                if actor == picked_actor:
                    print(f"Picked move axis: {axis}")
                    return axis
        
        return None

class RotateGizmo:
    def __init__(self, renderer):
        self.renderer = renderer
        self.actors = {}
        self.is_visible = False
        self.selected_actor = None
        self.gizmo_radius = 4.0  # Increased radius
        
        # Store original actor properties for opacity restoration
        self.original_opacities = {}
        
        # Create the gizmo circles
        self.create_gizmo()
    
    def create_gizmo(self):
        """Create the rotate gizmo with X, Y, Z circles - LARGER SIZE"""
        # X-axis (Red) - circle in YZ plane
        x_circle = self.create_circle([1, 0, 0], [0, 0, 0], self.gizmo_radius, [1.0, 0.0, 0.0])
        # Y-axis (Green) - circle in XZ plane
        y_circle = self.create_circle([0, 1, 0], [0, 0, 0], self.gizmo_radius, [0.0, 1.0, 0.0])
        # Z-axis (Blue) - circle in XY plane
        z_circle = self.create_circle([0, 0, 1], [0, 0, 0], self.gizmo_radius, [0.0, 0.4, 1.0])
        
        self.actors['x'] = x_circle
        self.actors['y'] = y_circle
        self.actors['z'] = z_circle
        
    
    def create_circle(self, normal, center, radius, color):
        """Create a circle actor for rotation - THICKER AND MORE VISIBLE"""
        # Use tube filter to create thicker, more visible circles
        circle_source = vtk.vtkRegularPolygonSource()
        circle_source.SetNumberOfSides(64)  # Smooth circle
        circle_source.SetRadius(radius)
        circle_source.SetCenter(center)
        circle_source.SetNormal(normal)
        
        # Create tube filter to make the circle thicker
        tube_filter = vtk.vtkTubeFilter()
        tube_filter.SetInputConnection(circle_source.GetOutputPort())
        tube_filter.SetRadius(0.25)  # Much thicker circles
        tube_filter.SetNumberOfSides(12)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tube_filter.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetOpacity(0.9)  # Make them more opaque
        actor.GetProperty().SetLineWidth(4)  # Additional line width
        
        return actor
    
    def show(self, target_actor):
        """Show gizmo on target actor and reduce object opacity"""
        print(f"RotateGizmo.show() called with actor: {target_actor}")
        self.selected_actor = target_actor
        if target_actor:
            # Store original opacity and reduce it more for rotation
            original_opacity = target_actor.GetProperty().GetOpacity()
            self.original_opacities[target_actor] = original_opacity
            target_actor.GetProperty().SetOpacity(0.2)  # Even more transparent for rotation
            print(f"Reduced opacity from {original_opacity} to 0.2")
            
            # Get target position
            position = target_actor.GetPosition()
            print(f"Positioning rotate gizmo at: {position}")
            
            # Position gizmo at target
            for axis_name, axis_actor in self.actors.items():
                axis_actor.SetPosition(position)
                self.renderer.AddActor(axis_actor)
                print(f"Added {axis_name} rotate circle to renderer")
            
            self.is_visible = True
            print("Rotate gizmo is now visible")
            
            # Force render only if render window exists
            self.safe_render()
        else:
            print("RotateGizmo.show(): No target actor provided!")
    
    def hide(self):
        """Hide gizmo and restore object opacity"""
        print("RotateGizmo.hide() called")
        
        # Restore original opacities
        for actor, original_opacity in self.original_opacities.items():
            if actor and hasattr(actor, 'GetProperty'):  # Additional safety check
                actor.GetProperty().SetOpacity(original_opacity)
                print(f"Restored opacity to {original_opacity}")
        self.original_opacities.clear()
        
        # Remove gizmo actors
        for axis_name, axis_actor in self.actors.items():
            self.renderer.RemoveActor(axis_actor)
            print(f"Removed {axis_name} rotate circle from renderer")
        
        self.is_visible = False
        self.selected_actor = None
        
        # Force render only if render window exists
        self.safe_render()
    
    def safe_render(self):
        """Safely render only if render window exists"""
        try:
            render_window = self.renderer.GetRenderWindow()
            if render_window:
                render_window.Render()
        except Exception as e:
            print(f"Safe render failed: {e}")
    
    def update_position(self):
        """Update gizmo position to follow selected object"""
        if self.is_visible and self.selected_actor:
            position = self.selected_actor.GetPosition()
            for axis_actor in self.actors.values():
                axis_actor.SetPosition(position)
    
    def get_axis_at_position(self, x, y):
        """Check if gizmo axis is clicked at screen position"""
        if not self.is_visible:
            return None
        
        picker = vtk.vtkPropPicker()
        picker.Pick(x, y, 0, self.renderer)
        
        picked_actor = picker.GetActor()
        if picked_actor in self.actors.values():
            # Find which axis was picked
            for axis, actor in self.actors.items():
                if actor == picked_actor:
                    print(f"Picked rotate axis: {axis}")
                    return axis
        
        return None
    
class ScaleGizmo:
    def __init__(self, renderer):
        self.renderer = renderer
        self.actors = {}
        self.is_visible = False
        self.selected_actor = None
        self.gizmo_size = 5.0
        
        # Store original actor properties for restoration
        self.original_opacities = {}
        
        # Create the gizmo
        self.create_gizmo()
    
    def create_gizmo(self):
        """Create the scale gizmo with X, Y, Z handles (cubes instead of arrows)"""
        # X-axis (Red) - cube pointing right
        x_cube = self.create_cube_handle([self.gizmo_size, 0, 0], [1.0, 0.0, 0.0])
        # Y-axis (Green) - cube pointing forward  
        y_cube = self.create_cube_handle([0, self.gizmo_size, 0], [0.0, 1.0, 0.0])
        # Z-axis (Blue) - cube pointing up
        z_cube = self.create_cube_handle([0, 0, self.gizmo_size], [0.0, 0.4, 1.0])
        
        # Uniform scale handle (center cube, usually yellow in Blender)
        uniform_cube = self.create_uniform_handle([0, 0, 0], [1.0, 1.0, 0.0])
        
        self.actors['x'] = x_cube
        self.actors['y'] = y_cube
        self.actors['z'] = z_cube
        self.actors['uniform'] = uniform_cube
    
    def create_cube_handle(self, position, color):
        """Create a cube handle for scaling along a specific axis"""
        cube_source = vtk.vtkCubeSource()
        cube_source.SetXLength(0.8)
        cube_source.SetYLength(0.8)
        cube_source.SetZLength(0.8)
        cube_source.SetCenter(position)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cube_source.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetLineWidth(2)
        
        return actor
    
    def create_uniform_handle(self, position, color):
        """Create the uniform scale handle (center cube)"""
        cube_source = vtk.vtkCubeSource()
        cube_source.SetXLength(1.2)
        cube_source.SetYLength(1.2)
        cube_source.SetZLength(1.2)
        cube_source.SetCenter(position)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cube_source.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetLineWidth(2)
        
        return actor
    
    def show(self, target_actor):
        """Show gizmo on target actor and reduce object opacity"""
        print(f"ScaleGizmo.show() called with actor: {target_actor}")
        self.selected_actor = target_actor
        if target_actor:
            # Store original opacity and reduce it
            original_opacity = target_actor.GetProperty().GetOpacity()
            self.original_opacities[target_actor] = original_opacity
            target_actor.GetProperty().SetOpacity(0.3)
            print(f"Reduced opacity from {original_opacity} to 0.3")
            
            # Get target position
            position = target_actor.GetPosition()
            print(f"Positioning scale gizmo at: {position}")
            
            # Position gizmo at target
            for axis_name, axis_actor in self.actors.items():
                axis_actor.SetPosition(position)
                self.renderer.AddActor(axis_actor)
                print(f"Added {axis_name} scale handle to renderer")
            
            self.is_visible = True
            print("Scale gizmo is now visible")
            
            self.safe_render()
        else:
            print("ScaleGizmo.show(): No target actor provided!")
    
    def hide(self):
        """Hide gizmo and restore object opacity"""
        print("ScaleGizmo.hide() called")
        
        # Restore original opacities
        for actor, original_opacity in self.original_opacities.items():
            if actor:
                actor.GetProperty().SetOpacity(original_opacity)
                print(f"Restored opacity to {original_opacity}")
        self.original_opacities.clear()
        
        # Remove gizmo actors
        for axis_name, axis_actor in self.actors.items():
            self.renderer.RemoveActor(axis_actor)
            print(f"Removed {axis_name} scale handle from renderer")
        
        self.is_visible = False
        self.selected_actor = None
        
        self.safe_render()
    
    def safe_render(self):
        """Safely render only if render window exists"""
        try:
            render_window = self.renderer.GetRenderWindow()
            if render_window:
                render_window.Render()
        except Exception as e:
            print(f"Safe render failed: {e}")
    
    def update_position(self):
        """Update gizmo position to follow selected object"""
        if self.is_visible and self.selected_actor:
            position = self.selected_actor.GetPosition()
            for axis_actor in self.actors.values():
                axis_actor.SetPosition(position)
    
    def get_axis_at_position(self, x, y):
        """Check if gizmo handle is clicked at screen position"""
        if not self.is_visible:
            return None
        
        picker = vtk.vtkPropPicker()
        picker.Pick(x, y, 0, self.renderer)
        
        picked_actor = picker.GetActor()
        if picked_actor in self.actors.values():
            # Find which handle was picked
            for axis, actor in self.actors.items():
                if actor == picked_actor:
                    print(f"Picked scale axis: {axis}")
                    return axis
        
        return None
    
class LeftToolbar(QToolBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Tools")
        self.setFixedWidth(60)
        self.setMovable(False)
        
        self.setOrientation(Qt.Vertical)
        self.setStyleSheet("""
            QToolBar {
                background-color: #323232;
                border: none;
                spacing: 2px;
                padding: 5px;
            }
            QToolButton {
                background-color: #323232;
                border: 1px solid #505050;
                border-radius: 3px;
                margin: 1px;
                padding: 5px;
            }
            QToolButton:hover {
                background-color: #404040;
                border: 1px solid #606060;
            }
            QToolButton:checked {
                background-color: #505050;
                border: 1px solid #ff8000;
            }
            QToolButton:pressed {
                background-color: #606060;
            }
        """)
        
        self.tools = {}
        self.current_tool = 'select'
        self.parent_widget = None
        
        self.create_tools()
    
    def create_tools(self):
        """Create the tool buttons with proper icons"""
        # Select/Move Tool (Arrow)
        select_btn = QToolButton(self)
        select_btn.setIcon(self.create_select_icon())
        select_btn.setToolTip("Select/Move Tool\n(LMB: Select object)")
        select_btn.setCheckable(True)
        select_btn.setChecked(True)
        select_btn.clicked.connect(lambda: self.set_tool('select'))
        select_btn.setFixedSize(48, 48)
        self.addWidget(select_btn)
        self.tools['select'] = select_btn
        
        # Box Select Tool
        box_select_btn = QToolButton(self)
        box_select_btn.setIcon(self.create_box_select_icon())
        box_select_btn.setToolTip("Box Select Tool\n(Drag: Select multiple objects)")
        box_select_btn.setCheckable(True)
        box_select_btn.clicked.connect(lambda: self.set_tool('box_select'))
        box_select_btn.setFixedSize(48, 48)
        self.addWidget(box_select_btn)
        self.tools['box_select'] = box_select_btn
        
        # Move Tool (NEW)
        move_btn = QToolButton(self)
        move_btn.setIcon(self.create_move_icon())
        move_btn.setToolTip("Move Tool\n(Drag: Move selected object)")
        move_btn.setCheckable(True)
        move_btn.clicked.connect(lambda: self.set_tool('move'))
        move_btn.setFixedSize(48, 48)
        self.addWidget(move_btn)
        self.tools['move'] = move_btn
        
        # Rotate Tool
        rotate_btn = QToolButton(self)
        rotate_btn.setIcon(self.create_rotate_icon())
        rotate_btn.setToolTip("Rotate Tool\n(Drag: Rotate selected object)")
        rotate_btn.setCheckable(True)
        rotate_btn.clicked.connect(lambda: self.set_tool('rotate'))
        rotate_btn.setFixedSize(48, 48)
        self.addWidget(rotate_btn)
        self.tools['rotate'] = rotate_btn
        
        # Scale Tool
        scale_btn = QToolButton(self)
        scale_btn.setIcon(self.create_scale_icon())
        scale_btn.setToolTip("Scale Tool\n(Drag: Scale selected object)")
        scale_btn.setCheckable(True)
        scale_btn.clicked.connect(lambda: self.set_tool('scale'))
        scale_btn.setFixedSize(48, 48)
        self.addWidget(scale_btn)
        self.tools['scale'] = scale_btn
        
        # MEASUREMENT TOOL - ADD THIS
        measure_btn = QToolButton(self)
        measure_btn.setIcon(self.create_measure_icon())
        measure_btn.setToolTip("Measurement Tool\n(LMB: Place points, Drag: Move points)")
        measure_btn.setCheckable(True)
        measure_btn.clicked.connect(lambda: self.set_tool('measure'))
        measure_btn.setFixedSize(48, 48)
        self.addWidget(measure_btn)
        self.tools['measure'] = measure_btn
        
        # Add spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.addWidget(spacer)
    
    def create_select_icon(self):
        """Create a proper select/move arrow icon"""
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw arrow
        painter.setPen(QPen(QColor(255, 255, 255), 3))
        painter.setBrush(QColor(255, 255, 255))
        
        # Arrow points (centered)
        arrow_points = [
            QPoint(16, 6),   # Tip
            QPoint(10, 20),  # Bottom left
            QPoint(14, 20),  # Notch left
            QPoint(14, 26),  # Stem left
            QPoint(18, 26),  # Stem right
            QPoint(18, 20),  # Notch right
            QPoint(22, 20)   # Bottom right
        ]
        
        painter.drawPolygon(arrow_points)
        painter.end()
        
        return QIcon(pixmap)
    
    def create_box_select_icon(self):
        """Create a proper box select icon"""
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw dashed rectangle
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.setBrush(Qt.NoBrush)
        
        # Main rectangle
        painter.drawRect(8, 8, 16, 16)
        
        # Draw crosshair/cursor in center
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.drawLine(16, 12, 16, 8)   # Top line
        painter.drawLine(20, 16, 24, 16)  # Right line
        painter.drawLine(16, 20, 16, 24)  # Bottom line
        painter.drawLine(12, 16, 8, 16)   # Left line
        
        painter.end()
        
        return QIcon(pixmap)
    
    def create_move_icon(self):
        """Create move tool icon (4-way arrow)"""
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        
        # Draw 4-way arrow
        center_x, center_y = 16, 16
        arrow_size = 6
        
        # Right arrow
        painter.drawLine(center_x, center_y, center_x + arrow_size, center_y)
        painter.drawLine(center_x + arrow_size, center_y, center_x + arrow_size - 3, center_y - 3)
        painter.drawLine(center_x + arrow_size, center_y, center_x + arrow_size - 3, center_y + 3)
        
        # Left arrow  
        painter.drawLine(center_x, center_y, center_x - arrow_size, center_y)
        painter.drawLine(center_x - arrow_size, center_y, center_x - arrow_size + 3, center_y - 3)
        painter.drawLine(center_x - arrow_size, center_y, center_x - arrow_size + 3, center_y + 3)
        
        # Up arrow
        painter.drawLine(center_x, center_y, center_x, center_y - arrow_size)
        painter.drawLine(center_x, center_y - arrow_size, center_x - 3, center_y - arrow_size + 3)
        painter.drawLine(center_x, center_y - arrow_size, center_x + 3, center_y - arrow_size + 3)
        
        # Down arrow
        painter.drawLine(center_x, center_y, center_x, center_y + arrow_size)
        painter.drawLine(center_x, center_y + arrow_size, center_x - 3, center_y + arrow_size - 3)
        painter.drawLine(center_x, center_y + arrow_size, center_x + 3, center_y + arrow_size - 3)
        
        painter.end()
        return QIcon(pixmap)
    
    def create_rotate_icon(self):
        """Create rotate tool icon (circular arrow)"""
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        
        # Draw circular arrow
        center_x, center_y = 16, 16
        radius = 10
        
        # Draw circle
        painter.drawEllipse(center_x - radius, center_y - radius, radius * 2, radius * 2)
        
        # Draw arrow head
        arrow_size = 4
        angle = math.radians(45)
        arrow_x = center_x + radius * math.cos(angle)
        arrow_y = center_y + radius * math.sin(angle)
        
        # Arrow points
        painter.drawLine(int(arrow_x), int(arrow_y), 
                        int(arrow_x - arrow_size), int(arrow_y - arrow_size))
        painter.drawLine(int(arrow_x), int(arrow_y),
                        int(arrow_x - arrow_size), int(arrow_y + arrow_size))
        
        painter.end()
        return QIcon(pixmap)
    
    def create_scale_icon(self):
        """Create scale tool icon (four arrows pointing outwards)"""
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        
        center_x, center_y = 16, 16
        arrow_size = 6
        
        # Right arrow
        painter.drawLine(center_x, center_y, center_x + arrow_size, center_y)
        painter.drawLine(center_x + arrow_size, center_y, center_x + arrow_size - 3, center_y - 3)
        painter.drawLine(center_x + arrow_size, center_y, center_x + arrow_size - 3, center_y + 3)
        
        # Left arrow  
        painter.drawLine(center_x, center_y, center_x - arrow_size, center_y)
        painter.drawLine(center_x - arrow_size, center_y, center_x - arrow_size + 3, center_y - 3)
        painter.drawLine(center_x - arrow_size, center_y, center_x - arrow_size + 3, center_y + 3)
        
        # Up arrow
        painter.drawLine(center_x, center_y, center_x, center_y - arrow_size)
        painter.drawLine(center_x, center_y - arrow_size, center_x - 3, center_y - arrow_size + 3)
        painter.drawLine(center_x, center_y - arrow_size, center_x + 3, center_y - arrow_size + 3)
        
        # Down arrow
        painter.drawLine(center_x, center_y, center_x, center_y + arrow_size)
        painter.drawLine(center_x, center_y + arrow_size, center_x - 3, center_y + arrow_size - 3)
        painter.drawLine(center_x, center_y + arrow_size, center_x + 3, center_y + arrow_size - 3)
        
        painter.end()
        return QIcon(pixmap)
    
    def create_measure_icon(self):
        """Create measurement tool icon (ruler)"""
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        
        # Draw ruler shape
        start_x, start_y = 8, 20
        end_x, end_y = 24, 12
        
        # Main ruler line
        painter.drawLine(start_x, start_y, end_x, end_y)
        
        # Measurement ticks
        tick_length = 4
        for i in range(0, 5):
            t = i / 4.0
            tick_x = start_x + (end_x - start_x) * t
            tick_y = start_y + (end_y - start_y) * t
            
            # Calculate perpendicular direction for ticks
            dx = end_x - start_x
            dy = end_y - start_y
            length = math.sqrt(dx*dx + dy*dy)
            if length > 0:
                perp_x = -dy / length
                perp_y = dx / length
                
                tick_end_x = tick_x + perp_x * tick_length
                tick_end_y = tick_y + perp_y * tick_length
                
                painter.drawLine(int(tick_x), int(tick_y), int(tick_end_x), int(tick_end_y))
        
        painter.end()
        return QIcon(pixmap)
    
    def set_tool(self, tool_name):
        """Set the current active tool"""
        if tool_name == self.current_tool:
            return
            
        # Uncheck previous tool
        if self.current_tool in self.tools:
            self.tools[self.current_tool].setChecked(False)
        
        # Check new tool
        self.tools[tool_name].setChecked(True)
        self.current_tool = tool_name
        
        print(f"Tool changed to: {tool_name}")
        
        # Notify parent
        if self.parent_widget:
            self.parent_widget.on_tool_changed(tool_name)

class RightPanel(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Properties", parent)
        self.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.setMinimumWidth(220)  # Slightly smaller minimum width
        
        # Create content for right panel
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(4, 4, 4, 4)  # Reduced margins
        layout.setSpacing(4)  # Reduced spacing
        
        # ===== SCENE LIST SECTION =====
        self.scene_header = QToolButton()
        self.scene_header.setText("Scene List ▼")
        self.scene_header.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.scene_header.setCheckable(True)
        self.scene_header.setChecked(True)
        self.scene_header.setStyleSheet("""
            QToolButton {
                background-color: #404040;
                color: white;
                border: 1px solid #505050;
                border-radius: 3px;
                padding: 5px;
                text-align: left;
                font-weight: bold;
            }
            QToolButton:checked {
                background-color: #505050;
            }
            QToolButton:hover {
                background-color: #484848;
            }
        """)
        layout.addWidget(self.scene_header)
        
        # Scene list content frame
        self.scene_content = QFrame()
        self.scene_content.setFrameStyle(QFrame.NoFrame)
        self.scene_content.setStyleSheet("background-color: #383838; border-radius: 3px; margin: 2px;")
        scene_layout = QVBoxLayout(self.scene_content)
        scene_layout.setContentsMargins(0, 0, 0, 0)
        scene_layout.setSpacing(0)
        
        # Scroll area for scene list
        self.scene_scroll_area = QScrollArea()
        self.scene_scroll_area.setWidgetResizable(True)
        self.scene_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scene_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scene_scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background-color: #2b2b2b;
                width: 12px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #505050;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #606060;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
        """)
        
        self.scene_list_widget = QWidget()
        self.scene_list_layout = QVBoxLayout(self.scene_list_widget)
        self.scene_list_layout.setContentsMargins(4, 4, 4, 4)
        self.scene_list_layout.setSpacing(2)
        self.scene_list_layout.addStretch()
        
        self.scene_scroll_area.setWidget(self.scene_list_widget)
        scene_layout.addWidget(self.scene_scroll_area)
        layout.addWidget(self.scene_content)
        
        # Toggle scene header
        self.scene_header.clicked.connect(self.toggle_scene_content)
        
        # Object info
        self.object_info = QLabel("No object selected")
        self.object_info.setAlignment(Qt.AlignCenter)
        info_font = QFont()
        info_font.setPointSize(9)
        self.object_info.setFont(info_font)
        self.object_info.setStyleSheet("color: #888; padding: 6px; background-color: #323232; border-radius: 3px; font-size: 9px;")
        self.object_info.setWordWrap(True)
        layout.addWidget(self.object_info)
        
        # Transform widget
        self.transform_widget = TransformWidget()
        layout.addWidget(self.transform_widget)
        
        # Geometry info section
        self.geometry_header = QToolButton()
        self.geometry_header.setText("Geometry Info ▼")
        self.geometry_header.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.geometry_header.setCheckable(True)
        self.geometry_header.setChecked(True)
        self.geometry_header.setStyleSheet("""
            QToolButton {
                background-color: #404040;
                color: white;
                border: 1px solid #505050;
                border-radius: 3px;
                padding: 5px;
                text-align: left;
                margin-top: 5px;
            }
            QToolButton:checked {
                background-color: #505050;
            }
            QToolButton:hover {
                background-color: #484848;
            }
        """)
        self.geometry_header.clicked.connect(self.toggle_geometry_content)
        layout.addWidget(self.geometry_header)
        
        self.geometry_content = QFrame()
        self.geometry_content.setFrameStyle(QFrame.NoFrame)
        self.geometry_content.setStyleSheet("background-color: #383838; border-radius: 3px; margin: 2px;")
        geometry_layout = QVBoxLayout(self.geometry_content)
        geometry_layout.setContentsMargins(8, 8, 8, 8)
        geometry_layout.setSpacing(4)
        
        self.vertices_label = QLabel("Vertices: -")
        self.faces_label = QLabel("Faces: -")
        self.edges_label = QLabel("Edges: -")
        self.corners_label = QLabel("Corners: -")
        self.bounds_label = QLabel("Bounds: -")
        
        geometry_labels_style = "color: #cccccc; background-color: transparent;"
        self.vertices_label.setStyleSheet(geometry_labels_style)
        self.faces_label.setStyleSheet(geometry_labels_style)
        self.edges_label.setStyleSheet(geometry_labels_style)
        self.corners_label.setStyleSheet(geometry_labels_style)
        self.bounds_label.setStyleSheet(geometry_labels_style)
        
        geometry_layout.addWidget(self.vertices_label)
        geometry_layout.addWidget(self.faces_label)
        geometry_layout.addWidget(self.edges_label)
        geometry_layout.addWidget(self.corners_label)
        geometry_layout.addWidget(self.bounds_label)
        
        layout.addWidget(self.geometry_content)

        # NEW: Lighting widget (Option B)
        self.lighting_widget = LightingWidget()
        layout.addWidget(self.lighting_widget)
        
        self.camera_header = QToolButton()
        self.camera_header.setText("Camera Controls ▼")
        self.camera_header.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.camera_header.setCheckable(True)
        self.camera_header.setChecked(True)
        self.camera_header.setStyleSheet("""
            QToolButton {
                background-color: #404040;
                color: white;
                border: 1px solid #505050;
                border-radius: 3px;
                padding: 5px;
                text-align: left;
                margin-top: 5px;
            }
            QToolButton:checked {
                background-color: #505050;
            }
            QToolButton:hover {
                background-color: #484848;
            }
        """)
        self.camera_header.clicked.connect(self.toggle_camera_content)
        layout.addWidget(self.camera_header)
        
        # Camera content frame
        self.camera_content = QFrame()
        self.camera_content.setFrameStyle(QFrame.NoFrame)
        self.camera_content.setStyleSheet("background-color: #383838; border-radius: 3px; margin: 2px;")
        camera_layout = QVBoxLayout(self.camera_content)
        camera_layout.setContentsMargins(8, 8, 8, 8)
        camera_layout.setSpacing(4)
        
         # Toggle camera view button
        self.toggle_camera_view_btn = QPushButton("Toggle Camera View: OFF")
        self.toggle_camera_view_btn.setCheckable(True)
        self.toggle_camera_view_btn.setChecked(False)
        self.toggle_camera_view_btn.setStyleSheet("""
            QPushButton {
                background-color: #505050;
                color: white;
                border: 1px solid #606060;
                border-radius: 3px;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: #585858;
            }
            QPushButton:pressed {
                background-color: #404040;
            }
            QPushButton:checked {
                background-color: #ff8000;
                color: black;
            }
            QPushButton:disabled {
                background-color: #383838;
                color: #666666;
            }
        """)
        self.toggle_camera_view_btn.clicked.connect(self.toggle_camera_view)
        camera_layout.addWidget(self.toggle_camera_view_btn)
        
        # Save camera view button (keep this one)
        self.save_camera_view_btn = QPushButton("Save Camera View as Image")
        self.save_camera_view_btn.setStyleSheet("""
            QPushButton {
                background-color: #505050;
                color: white;
                border: 1px solid #606060;
                border-radius: 3px;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: #585858;
            }
            QPushButton:pressed {
                background-color: #404040;
            }
            QPushButton:disabled {
                background-color: #383838;
                color: #666666;
            }
        """)
        self.save_camera_view_btn.clicked.connect(self.save_camera_view)
        camera_layout.addWidget(self.save_camera_view_btn)
        
        # Spacer to push stuff up
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(spacer)
        
        # Wrap everything in a scroll area so it becomes scrollable if too full
        outer_scroll = QScrollArea()
        outer_scroll.setWidgetResizable(True)
        outer_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        outer_scroll.setStyleSheet("""
            QScrollArea {
                background-color: transparent;
                border: none;
            }
            QScrollArea > QWidget {
                background-color: #2b2b2b;
            }
            QScrollArea > QWidget > QWidget {
                background-color: #2b2b2b;
            }
        """)
        outer_scroll.setWidget(content)
        self.setWidget(outer_scroll)
        
        # Styling
        self.setStyleSheet("""
            QDockWidget {
                background-color: #2b2b2b;
                color: white;
                titlebar-normal-icon: url(none);
                titlebar-close-icon: url(none);
                font-size: 9px;
            }
            QDockWidget::title {
                background-color: #323232;
                padding: 3px;
                text-align: center;
                font-size: 9px;
            }
        """)
        
        # Scene list mapping
        self.scene_items = {}   # actor -> UI data
        self.selected_scene_item = None
        
        # Timer to keep panel synced
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_panel)
        self.update_timer.start(100)
        
    def toggle_scene_content(self):
        """Toggle scene list visibility"""
        if self.scene_header.isChecked():
            self.scene_content.show()
            self.scene_header.setText("Scene List ▼")
        else:
            self.scene_content.hide()
            self.scene_header.setText("Scene List ▶")
        
    def toggle_geometry_content(self):
        """Toggle geometry section visibility"""
        if self.geometry_header.isChecked():
            self.geometry_content.show()
            self.geometry_header.setText("Geometry Info ▼")
        else:
            self.geometry_content.hide()
            self.geometry_header.setText("Geometry Info ▶")
    
    def set_vtk_widget(self, vtk_widget):
        """Set reference to VTK widget"""
        self.transform_widget.vtk_widget = vtk_widget
        self.lighting_widget.set_vtk_widget(vtk_widget)
        self.vtk_widget = vtk_widget

    def update_panel(self):
        """Update both scene list and object information"""
        self.update_scene_list()
        self.update_object_info()
        
    def update_scene_list(self):
        """Update the scene list with current objects"""
        if not hasattr(self, 'vtk_widget') or not self.vtk_widget:
            return
            
        current_actors = set(self.vtk_widget.object_manager.actors)
        current_scene_items = set(self.scene_items.keys())
        
        # Remove deleted objects from scene list
        for actor in current_scene_items - current_actors:
            self.remove_scene_item(actor)
        
        # Add new objects to scene list
        for actor in current_actors - current_scene_items:
            self.add_scene_item(actor)
        
        # Update selection highlighting
        self.update_scene_selection()
    
    def add_scene_item(self, actor):
        """Add an object to the scene list"""
        if actor in self.scene_items:
            return
            
        # Create scene item widget
        item_widget = QWidget()
        item_layout = QHBoxLayout(item_widget)
        item_layout.setContentsMargins(6, 4, 6, 4)
        item_layout.setSpacing(6)
        
        # Object icon (small colored square)
        icon_label = QLabel()
        icon_label.setFixedSize(12, 12)
        color = actor.GetProperty().GetColor()
        icon_style = f"background-color: rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}); border-radius: 2px;"
        icon_label.setStyleSheet(icon_style)
        
        # Object name
        name_label = QLabel(self.get_object_name(actor))
        name_label.setStyleSheet("color: #cccccc; background: transparent;")
        name_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        
        # Visibility toggle button
        visibility_btn = QToolButton()
        visibility_btn.setText("●")  # Eye symbol alternative
        visibility_btn.setToolTip("Toggle visibility")
        visibility_btn.setFixedSize(20, 20)
        visibility_btn.setStyleSheet("""
            QToolButton {
                color: #00ff00;
                background: transparent;
                border: none;
                font-size: 10px;
            }
            QToolButton:hover {
                color: #80ff80;
            }
        """)
        visibility_btn.setCheckable(True)
        visibility_btn.setChecked(True)
        visibility_btn.clicked.connect(lambda checked, a=actor: self.toggle_object_visibility(a, checked))
        
        # Add widgets to layout
        item_layout.addWidget(icon_label)
        item_layout.addWidget(name_label)
        item_layout.addWidget(visibility_btn)
        
        # Set item widget style
        item_widget.setStyleSheet("""
            QWidget {
                background-color: #323232;
                border-radius: 3px;
            }
            QWidget:hover {
                background-color: #3a3a3a;
            }
        """)
        
        # Make the item clickable
        item_widget.mousePressEvent = lambda event, a=actor: self.select_object_from_scene(a)
        
        # Insert before the stretch at the bottom
        self.scene_list_layout.insertWidget(self.scene_list_layout.count() - 1, item_widget)
        
        # Store reference
        self.scene_items[actor] = {
            'widget': item_widget,
            'name_label': name_label,
            'visibility_btn': visibility_btn
        }
    
    def remove_scene_item(self, actor):
        """Remove an object from the scene list"""
        if actor in self.scene_items:
            item_data = self.scene_items[actor]
            item_data['widget'].setParent(None)
            item_data['widget'].deleteLater()
            del self.scene_items[actor]
            
    def get_object_name(self, actor):
        """Generate a name for the object based on its type"""
        # First try to get the stored object type
        if hasattr(actor, '_object_type'):
            obj_type = actor._object_type
        # Then try the object manager's method
        elif hasattr(self, 'vtk_widget') and self.vtk_widget and hasattr(self.vtk_widget.object_manager, 'get_object_type'):
            obj_type = self.vtk_widget.object_manager.get_object_type(actor)
        else:
            obj_type = "Object"
        
        # Create a simple name with index
        index = list(self.scene_items.keys()).index(actor) if actor in self.scene_items else len(self.scene_items)
        return f"{obj_type} {index + 1}"
    
    def toggle_object_visibility(self, actor, visible):
        """Toggle object visibility"""
        if actor:
            actor.SetVisibility(visible)
            # Update visibility button
            if actor in self.scene_items:
                btn = self.scene_items[actor]['visibility_btn']
                btn.setText("●" if visible else "○")
                btn.setStyleSheet(f"""
                    QToolButton {{
                        color: {'#00ff00' if visible else '#666666'};
                        background: transparent;
                        border: none;
                        font-size: 10px;
                    }}
                    QToolButton:hover {{
                        color: {'#80ff80' if visible else '#888888'};
                    }}
                """)
            self.vtk_widget.render_window.Render()
    
    def select_object_from_scene(self, actor):
        """Select object when clicked in scene list"""
        if actor and self.vtk_widget:
            # Clear current selection
            self.vtk_widget.object_manager.deselect_all()
            
            # Select the clicked object
            self.vtk_widget.object_manager.select_object(actor)
            
            # Update scene list selection
            self.update_scene_selection()
            
            # Force render
            self.vtk_widget.render_window.Render()
    
    def update_scene_selection(self):
        """Update selection highlighting in scene list"""
        selected_actors = set(self.vtk_widget.object_manager.selected_actors) if self.vtk_widget else set()
        
        for actor, item_data in self.scene_items.items():
            is_selected = actor in selected_actors
            style = """
                QWidget {
                    background-color: #404040;
                    border: 2px solid #ff8000;
                    border-radius: 3px;
                }
            """ if is_selected else """
                QWidget {
                    background-color: #323232;
                    border: none;
                    border-radius: 3px;
                }
                QWidget:hover {
                    background-color: #3a3a3a;
                }
            """
            item_data['widget'].setStyleSheet(style)
    
    def update_object_info(self):
        """Update object information display with comprehensive geometry data"""
        if not hasattr(self, 'transform_widget') or not self.transform_widget.vtk_widget:
            return
            
        obj_manager = self.transform_widget.vtk_widget.object_manager
        if obj_manager.selected_actors:
            selected_actor = obj_manager.selected_actors[0]
            position = selected_actor.GetPosition()
            orientation = selected_actor.GetOrientation()
            scale = selected_actor.GetScale()
            
            # Get geometry information
            geometry_info = self.get_geometry_info(selected_actor)
            
            # Update main object info
            info_text = f"Selected Object\n"
            info_text += f"Position: X={position[0]:.2f} Y={position[1]:.2f} Z={position[2]:.2f}\n"
            info_text += f"Rotation: X={orientation[0]:.1f}° Y={orientation[1]:.1f}° Z={orientation[2]:.1f}°\n"
            info_text += f"Scale: X={scale[0]:.2f} Y={scale[1]:.2f} Z={scale[2]:.2f}"
            
            self.object_info.setText(info_text)
            self.object_info.setStyleSheet("color: #ffffff; padding: 6px; background-color: #404040; border-radius: 3px; font-size: 9px;")
            
            # Update geometry info
            self.vertices_label.setText(f"Vertices: {geometry_info['vertices']}")
            self.faces_label.setText(f"Faces: {geometry_info['faces']}")
            self.edges_label.setText(f"Edges: {geometry_info['edges']}")
            self.corners_label.setText(f"Corners: {geometry_info['corners']}")
            self.bounds_label.setText(f"Bounds: {geometry_info['bounds']}")
            
        else:
            self.object_info.setText("No object selected\n\nSelect an object to see\nits properties here")
            self.object_info.setStyleSheet("color: #888; padding: 6px; background-color: #323232; border-radius: 3px; font-size: 9px;")
            
            # Reset geometry info
            self.vertices_label.setText("Vertices: -")
            self.faces_label.setText("Faces: -")
            self.edges_label.setText("Edges: -")
            self.corners_label.setText("Corners: -")
            self.bounds_label.setText("Bounds: -")
            
    def get_geometry_info(self, actor):
        """Extract comprehensive geometry information from actor"""
        try:
            mapper = actor.GetMapper()
            if not mapper:
                return self.get_default_geometry_info()
            
            input_data = mapper.GetInput()
            if not input_data:
                return self.get_default_geometry_info()
            
            # Get points (vertices)
            points = input_data.GetPoints()
            num_vertices = points.GetNumberOfPoints() if points else 0
            
            # Get polygons (faces)
            polygons = input_data.GetPolys()
            num_faces = polygons.GetNumberOfCells() if polygons else 0
            
            # Calculate edges and corners
            num_edges = self.calculate_edge_count(input_data)
            num_corners = self.calculate_corner_count(input_data)
            
            # Get bounds
            bounds = input_data.GetBounds()
            bounds_str = f"X:{bounds[0]:.1f}-{bounds[1]:.1f} Y:{bounds[2]:.1f}-{bounds[3]:.1f} Z:{bounds[4]:.1f}-{bounds[5]:.1f}"
            
            return {
                'vertices': num_vertices,
                'faces': num_faces,
                'edges': num_edges,
                'corners': num_corners,
                'bounds': bounds_str
            }
            
        except Exception as e:
            print(f"Error getting geometry info: {e}")
            return self.get_default_geometry_info()
    
    def calculate_edge_count(self, polydata):
        """Calculate the number of unique edges in the mesh"""
        try:
            if not polydata:
                return 0
                
            edges = set()
            polygons = polydata.GetPolys()
            
            if not polygons:
                return 0
                
            polygons.InitTraversal()
            id_list = vtk.vtkIdList()
            
            while polygons.GetNextCell(id_list):
                num_points = id_list.GetNumberOfIds()
                
                # For each edge in the polygon
                for i in range(num_points):
                    point1 = id_list.GetId(i)
                    point2 = id_list.GetId((i + 1) % num_points)
                    
                    # Store edge as sorted tuple to avoid duplicates
                    edge = tuple(sorted([point1, point2]))
                    edges.add(edge)
            
            return len(edges)
            
        except Exception as e:
            print(f"Error calculating edges: {e}")
            return 0
    
    def calculate_corner_count(self, polydata):
        """Calculate the number of corners (unique points where edges meet)"""
        try:
            if not polydata:
                return 0
                
            # For polygonal meshes, corners are essentially the vertices
            points = polydata.GetPoints()
            return points.GetNumberOfPoints() if points else 0
            
        except Exception as e:
            print(f"Error calculating corners: {e}")
            return 0
    
    def get_default_geometry_info(self):
        """Return default values when geometry info can't be calculated"""
        return {
            'vertices': 0,
            'faces': 0,
            'edges': 0,
            'corners': 0,
            'bounds': 'Unknown'
        }
    
    def toggle_camera_content(self):
        """Toggle camera controls visibility"""
        if self.camera_header.isChecked():
            self.camera_content.show()
            self.camera_header.setText("Camera Controls ▼")
        else:
            self.camera_content.hide()
            self.camera_header.setText("Camera Controls ▶")
    
    def view_from_camera(self):
        """Set the view to the selected camera's perspective"""
        if not hasattr(self, 'vtk_widget') or not self.vtk_widget:
            return
            
        if not self.vtk_widget.object_manager.selected_actors:
            QMessageBox.warning(self, "No Selection", "Please select a camera first.")
            return
            
        selected_actor = self.vtk_widget.object_manager.selected_actors[0]
        
        # Check if the selected actor is a camera
        if not hasattr(selected_actor, '_is_camera') or not selected_actor._is_camera:
            QMessageBox.warning(self, "Not a Camera", "Please select a camera object.")
            return
            
        # Get the camera object
        camera_obj = selected_actor._camera_object
        if camera_obj:
            self.vtk_widget.set_camera_view(camera_obj)
    
    def toggle_camera_view(self, checked):
        """Toggle between camera view and main view"""
        if not hasattr(self, 'vtk_widget') or not self.vtk_widget:
            return
            
        if checked:
            # Switch to camera view
            if not self.vtk_widget.object_manager.selected_actors:
                QMessageBox.warning(self, "No Selection", "Please select a camera first.")
                self.toggle_camera_view_btn.setChecked(False)
                return
                
            selected_actor = self.vtk_widget.object_manager.selected_actors[0]
            
            # Check if the selected actor is a camera
            if not hasattr(selected_actor, '_is_camera') or not selected_actor._is_camera:
                QMessageBox.warning(self, "Not a Camera", "Please select a camera object.")
                self.toggle_camera_view_btn.setChecked(False)
                return
                
            # Get the camera object
            camera_obj = selected_actor._camera_object
            if camera_obj:
                success = self.vtk_widget.set_camera_view(camera_obj)
                if success:
                    self.toggle_camera_view_btn.setText("Toggle Camera View: ON")
                    print("Switched to camera view")
                else:
                    self.toggle_camera_view_btn.setChecked(False)
            else:
                self.toggle_camera_view_btn.setChecked(False)
        else:
            # Switch back to main view
            self.vtk_widget.reset_to_main_view()
            self.toggle_camera_view_btn.setText("Toggle Camera View: OFF")
            print("Switched to main view")
            
    def save_camera_view(self):
        """Save the current camera view as an image"""
        if not hasattr(self, 'vtk_widget') or not self.vtk_widget:
            return
            
        # Get file path for saving
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Camera View as Image",
            f"camera_view_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            "PNG Images (*.png);;JPEG Images (*.jpg *.jpeg);;All Files (*)"
        )
        
        if file_path:
            success = self.vtk_widget.save_view_image(file_path)
            if success:
                QMessageBox.information(self, "Success", f"Camera view saved as:\n{file_path}")
            else:
                QMessageBox.warning(self, "Error", "Failed to save camera view image.")
    
    def reset_to_main_view(self):
        """Reset to the main camera view"""
        if hasattr(self, 'vtk_widget') and self.vtk_widget:
            self.vtk_widget.reset_view()
            
class BoxSelectRubberBand(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setParent(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Use Tool instead of SubWindow to avoid stacking issues
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint)
        
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.is_selecting = False
        
        # Ensure completely transparent background
        self.setStyleSheet("background: transparent;")
        
        self.hide()
    
    def paintEvent(self, event):
        """Custom paint to create a clean dotted line selection box"""
        if not self.is_selecting:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw ONLY the dotted border, no background fill
        pen = QPen(QColor(255, 255, 255), 2)
        pen.setStyle(Qt.DashLine)
        pen.setDashPattern([3, 3])  # 3px dash, 3px gap
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)  # No fill
        
        # Draw the rectangle
        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.drawRect(rect)
        
        painter.end()
    
    def start_selection(self, start_point):
        """Start box selection - ensure clean state"""
        self.start_point = start_point
        self.end_point = start_point
        self.is_selecting = True
        self.update_geometry()
        self.show()
        self.raise_()
        self.update()  # Force repaint
    
    def update_selection(self, end_point):
        """Update selection box"""
        self.end_point = end_point
        self.update_geometry()
        self.update()  # Force repaint
    
    def end_selection(self):
        """End box selection"""
        self.is_selecting = False
        self.hide()
        self.update()  # Force repaint
    
    def update_geometry(self):
        """Update the rubber band geometry"""
        x = min(self.start_point.x(), self.end_point.x())
        y = min(self.start_point.y(), self.end_point.y())
        width = max(2, abs(self.start_point.x() - self.end_point.x()))
        height = max(2, abs(self.start_point.y() - self.end_point.y()))
        
        self.setGeometry(x, y, width, height)
        
class ObjectManager:
    def __init__(self, renderer):
        self.renderer = renderer
        self.actors = []
        self.mappers = []
        self.sources = []
        self.current_object = 'sphere'
        self.current_color = (1.0, 1.0, 1.0)
        self.selected_actors = []
        self.outline_actors = {}
        self.cameras = []
        
        # Create all gizmos
        self.move_gizmo = MoveGizmo(renderer)
        self.rotate_gizmo = RotateGizmo(renderer)
        self.scale_gizmo = ScaleGizmo(renderer)  # ADD THIS LINE
        self.active_gizmo = None
        
    def add_external_model(self, actor, mapper, source=None):
        """Add an externally loaded model to the object manager"""
        self.actors.append(actor)
        self.mappers.append(mapper)
        if source:
            self.sources.append(source)
        
        # Create outline
        self.create_outline(actor)
        
        # Auto-select the new object
        self.select_object(actor)
        
        return actor
    
    def get_object_geometry_info(self, actor):
        """Get geometry information for a specific actor"""
        try:
            mapper = actor.GetMapper()
            if not mapper:
                return None
                
            # Get the polydata from the mapper
            polydata = mapper.GetInput()
            if not polydata:
                return None
                
            return polydata
            
        except Exception as e:
            print(f"Error getting object geometry: {e}")
            return None
        
    def create_object(self, object_type):
        """Create a 3D geometric object at origin"""
        print(f"Creating {object_type} object...")
        
        source = None
        
        # Create geometric source at ORIGIN (0,0,0)
        if object_type == 'sphere':
            source = vtk.vtkSphereSource()
            source.SetCenter(0, 0, 0)  # Always at origin
            source.SetRadius(2.0)
            source.SetPhiResolution(32)
            source.SetThetaResolution(32)
        elif object_type == 'cube':
            source = vtk.vtkCubeSource()
            source.SetCenter(0, 0, 0)  # Always at origin
            source.SetXLength(3)
            source.SetYLength(3)
            source.SetZLength(3)
        elif object_type == 'cylinder':
            source = vtk.vtkCylinderSource()
            source.SetCenter(0, 0, 0)  # Always at origin
            source.SetHeight(3)
            source.SetRadius(1.5)
            source.SetResolution(32)
        elif object_type == 'cone':
            source = vtk.vtkConeSource()
            source.SetCenter(0, 0, 0)  # Always at origin
            source.SetHeight(3)
            source.SetRadius(1.5)
            source.SetResolution(32)
        elif object_type == 'pyramid':
            source = self.create_pyramid_source()
        elif object_type == 'torus':
            # You might need to implement this or use parametric source
            source = self.create_torus_source(2.0, 0.5)
        
            # Platonic solids
        elif object_type == 'tetrahedron':
            source = vtk.vtkPlatonicSolidSource()
            source.SetSolidTypeToTetrahedron()
        elif object_type == 'octahedron':
            source = vtk.vtkPlatonicSolidSource()
            source.SetSolidTypeToOctahedron()
        elif object_type == 'dodecahedron':
            source = vtk.vtkPlatonicSolidSource()
            source.SetSolidTypeToDodecahedron()
        elif object_type == 'icosahedron':
            source = vtk.vtkPlatonicSolidSource()
            source.SetSolidTypeToIcosahedron()
            
            # Parametric Objects
        elif object_type == 'mobius':
            source = vtk.vtkParametricMobius()
            source.SetRadius(2.0)
            source.SetMinimumV(-0.5)
            source.SetMaximumV(0.5)
            param_source = vtk.vtkParametricFunctionSource()
            param_source.SetParametricFunction(source)
            source = param_source
        elif object_type == 'klein':
            source = self.create_klein_bottle()
        elif object_type == 'super_toroid':
            source = self.create_super_toroid()
        elif object_type == 'super_ellipsoid':
            source = self.create_super_ellipsoid()
            
            # Cell Based Objects
        elif object_type == 'convex_point':
            source = self.create_convex_point_set()
        elif object_type == 'voxel':
            source = self.create_voxel_source()
        elif object_type == 'hexahedron':
            source = self.create_hexahedron_source()  
        elif object_type == 'polyhedron':
            source = self.create_polyhedron_source()
            
            # Isosurface Objects
        elif object_type == 'gyroid':
            source = self.create_gyroid_isosurface()
        elif object_type == 'schwarz_primitive':
            source = self.create_schwarz_primitive_isosurface()
        elif object_type == 'schwarz_diamond':
            source = self.create_schwarz_diamond_isosurface()
        elif object_type == 'schoen_iwp':
            source = self.create_schoen_iwp_isosurface()
        elif object_type == 'fischer_koch':
            source = self.create_fischer_koch_isosurface()
            
        elif object_type == 'camera':
            return self.create_camera()
            
        else:
            print(f"Unknown object type: {object_type}")
            return None
        
        if source:
        
            # Ensure the source is updated before creating the actor
            if hasattr(source, 'Update'):
                source.Update()
            
            return self.finalize_object_creation(source, object_type)
        
        return None
    
    def finalize_object_creation(self, source, object_type):
        """Finalize the object creation process - UPDATED to ensure geometry data"""
        self.sources.append(source)
        
        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
        
        # Ensure the mapper has data
        if hasattr(source, 'Update'):
            source.Update()
        
        self.mappers.append(mapper)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(self.current_color)
        actor.GetProperty().SetOpacity(1.0)
        actor.SetPosition(0, 0, 0)
        # Store object type directly on the actor for easy retrieval
        actor._object_type = self.get_detailed_object_type(object_type)  # ADD THIS LINE
        self.actors.append(actor)

        self.current_object = object_type
        
        # Add to renderer
        self.renderer.AddActor(actor)
        
        # Create outline
        self.create_outline(actor)
        
        # Auto-select the newly created object
        self.select_object(actor)
        
        return actor
    
    def get_detailed_object_type(self, creation_key):
        """Map creation keys to human-readable object types"""
        type_map = {
            # Geometric Objects
            'sphere': 'Sphere',
            'cube': 'Cube',
            'pyramid': 'Pyramid',
            'torus': 'Torus',
            'cylinder': 'Cylinder',
            'cone': 'Cone',
            
            # Cell Based Objects
            'convex_point': 'Convex Point Set',
            'voxel': 'Voxel',
            'hexahedron': 'Hexahedron',
            'polyhedron': 'Polyhedron',
            
            # Source Formats (Platonic Solids)
            'tetrahedron': 'Tetrahedron',
            'octahedron': 'Octahedron',
            'dodecahedron': 'Dodecahedron',
            'icosahedron': 'Icosahedron',
            
            # Parametric Objects
            'klein': 'Klein Bottle',
            'mobius': 'Mobius Strip',
            'super_toroid': 'Super Toroid',
            'super_ellipsoid': 'Super Ellipsoid',
            
            # Isosurface Objects
            'gyroid': 'Gyroid',
            'schwarz_primitive': 'Schwarz Primitive',
            'schwarz_diamond': 'Schwarz Diamond',
            'schoen_iwp': 'Schoen IWP',
            'fischer_koch': 'Fischer Koch S'
        }
        
        return type_map.get(creation_key, 'Object')
    
    def create_gyroid_isosurface(self):
        """Create a gyroid isosurface"""
        return self.create_periodic_isosurface("gyroid")
    
    def create_schwarz_primitive_isosurface(self):
        """Create a Schwarz Primitive isosurface"""
        return self.create_periodic_isosurface("schwarz_primitive")
    
    def create_schwarz_diamond_isosurface(self):
        """Create a Schwarz Diamond isosurface"""
        return self.create_periodic_isosurface("schwarz_diamond")
    
    def create_schoen_iwp_isosurface(self):
        """Create a Schoen I-WP isosurface"""
        return self.create_periodic_isosurface("schoen_iwp")
    
    def create_fischer_koch_isosurface(self):
        """Create a Fischer-Koch S isosurface"""
        return self.create_periodic_isosurface("fischer_koch")
    
    def create_periodic_isosurface(self, surface_type):
        """Create periodic minimal surfaces using mathematical functions"""
        # Create a 3D grid
        grid = vtk.vtkImageData()
        grid.SetDimensions(50, 50, 50)
        grid.SetSpacing(0.2, 0.2, 0.2)
        grid.SetOrigin(-5, -5, -5)
        
        # Add scalar data based on surface type
        scalars = vtk.vtkFloatArray()
        scalars.SetNumberOfComponents(1)
        scalars.SetName("Surface")
        
        for z in range(50):
            for y in range(50):
                for x in range(50):
                    px = x * 0.2 - 5
                    py = y * 0.2 - 5
                    pz = z * 0.2 - 5
                    
                    if surface_type == "gyroid":
                        value = (math.cos(px) * math.sin(py) + 
                                math.cos(py) * math.sin(pz) + 
                                math.cos(pz) * math.sin(px))
                    elif surface_type == "schwarz_primitive":
                        value = (math.cos(px) + math.cos(py) + math.cos(pz))
                    elif surface_type == "schwarz_diamond":
                        value = (math.sin(px) * math.sin(py) * math.sin(pz) +
                                math.sin(px) * math.cos(py) * math.cos(pz) +
                                math.cos(px) * math.sin(py) * math.cos(pz) +
                                math.cos(px) * math.cos(py) * math.sin(pz))
                    elif surface_type == "schoen_iwp":
                        value = (2*(math.cos(px)*math.cos(py) + math.cos(py)*math.cos(pz) + math.cos(pz)*math.cos(px)) -
                                (math.cos(2*px) + math.cos(2*py) + math.cos(2*pz)))
                    elif surface_type == "fischer_koch":
                        # Simplified Fischer-Koch S approximation
                        value = (math.cos(2*px) * math.cos(pz) +
                                math.cos(2*py) * math.cos(pz) +
                                math.cos(2*pz) * math.cos(px))
                    else:
                        value = 0
                    
                    scalars.InsertNextValue(value)
        
        grid.GetPointData().SetScalars(scalars)
        
        # Create isosurface
        contour = vtk.vtkContourFilter()
        contour.SetInputData(grid)
        contour.SetValue(0, 0.0)  # Isosurface at value 0
        
        return contour
    
    def create_klein_bottle(self):
        """Create a Klein bottle parametric surface"""
        klein = vtk.vtkParametricKlein()
        source = vtk.vtkParametricFunctionSource()
        source.SetParametricFunction(klein)
        source.SetUResolution(50)
        source.SetVResolution(50)
        return source
    
    def create_super_toroid(self):
        """Create a super toroid (modified torus)"""
        # Using super toroid parametric function
        super_toroid = vtk.vtkParametricSuperToroid()
        super_toroid.SetN1(0.5)  # Shape parameters
        super_toroid.SetN2(0.5)
        source = vtk.vtkParametricFunctionSource()
        source.SetParametricFunction(super_toroid)
        source.SetUResolution(50)
        source.SetVResolution(30)
        return source
    
    def create_super_ellipsoid(self):
        """Create a super ellipsoid"""
        super_ellipsoid = vtk.vtkParametricSuperEllipsoid()
        super_ellipsoid.SetN1(0.5)
        super_ellipsoid.SetN2(0.5)
        source = vtk.vtkParametricFunctionSource()
        source.SetParametricFunction(super_ellipsoid)
        source.SetUResolution(50)
        source.SetVResolution(50)
        return source
    
    def create_convex_point_set(self):
        """Create a convex point set (convex hull of random points)"""
        # Create some random points
        points = vtk.vtkPoints()
        for i in range(20):
            points.InsertNextPoint(
                np.random.uniform(-1, 1),
                np.random.uniform(-1, 1), 
                np.random.uniform(-1, 1)
            )
        
        # Create convex hull
        convex_hull = vtk.vtkDelaunay3D()
        convex_hull.SetInputData(vtk.vtkPolyData())
        convex_hull.GetInput().SetPoints(points)
        
        # Extract surface
        surface = vtk.vtkDataSetSurfaceFilter()
        surface.SetInputConnection(convex_hull.GetOutputPort())
        return surface
    
    def create_voxel_source(self):
        """Create a voxel (3D pixel)"""
        points = vtk.vtkPoints()
        points.InsertNextPoint(0, 0, 0)
        points.InsertNextPoint(1, 0, 0)
        points.InsertNextPoint(0, 1, 0)
        points.InsertNextPoint(1, 1, 0)
        points.InsertNextPoint(0, 0, 1)
        points.InsertNextPoint(1, 0, 1)
        points.InsertNextPoint(0, 1, 1)
        points.InsertNextPoint(1, 1, 1)
        
        voxel = vtk.vtkVoxel()
        for i in range(8):
            voxel.GetPointIds().SetId(i, i)
        
        ugrid = vtk.vtkUnstructuredGrid()
        ugrid.SetPoints(points)
        ugrid.InsertNextCell(voxel.GetCellType(), voxel.GetPointIds())
        
        geometry = vtk.vtkGeometryFilter()
        geometry.SetInputData(ugrid)
        return geometry
    
    def create_hexahedron_source(self):
        """Create a hexahedron (arbitrary 8-point 3D shape)"""
        points = vtk.vtkPoints()
        # Create a distorted cube
        points.InsertNextPoint(0, 0, 0)
        points.InsertNextPoint(2, 0, 0)
        points.InsertNextPoint(2, 1, 0)
        points.InsertNextPoint(0, 1, 0)
        points.InsertNextPoint(0, 0, 2)
        points.InsertNextPoint(2, 0, 2)
        points.InsertNextPoint(2, 1, 2)
        points.InsertNextPoint(0, 1, 2)
        
        hexahedron = vtk.vtkHexahedron()
        for i in range(8):
            hexahedron.GetPointIds().SetId(i, i)
        
        ugrid = vtk.vtkUnstructuredGrid()
        ugrid.SetPoints(points)
        ugrid.InsertNextCell(hexahedron.GetCellType(), hexahedron.GetPointIds())
        
        geometry = vtk.vtkGeometryFilter()
        geometry.SetInputData(ugrid)
        return geometry
    
    def create_polyhedron_source(self):
        """Create a polyhedron (complex 3D shape with multiple faces)"""
        # Create a triangular prism as a simple polyhedron
        points = vtk.vtkPoints()
        points.InsertNextPoint(0, 0, 0)
        points.InsertNextPoint(1, 0, 0)
        points.InsertNextPoint(0.5, 0.866, 0)  # Equilateral triangle
        points.InsertNextPoint(0, 0, 1)
        points.InsertNextPoint(1, 0, 1)
        points.InsertNextPoint(0.5, 0.866, 1)
        
        # Define faces (triangular prism has 5 faces: 2 triangles + 3 rectangles)
        faces = vtk.vtkIdList()
        
        # Bottom triangle
        faces.InsertNextId(3)  # 3 points
        faces.InsertNextId(0)
        faces.InsertNextId(1)
        faces.InsertNextId(2)
        
        # Top triangle
        faces.InsertNextId(3)
        faces.InsertNextId(3)
        faces.InsertNextId(4)
        faces.InsertNextId(5)
        
        # Rectangular sides
        faces.InsertNextId(4)
        faces.InsertNextId(0)
        faces.InsertNextId(1)
        faces.InsertNextId(4)
        faces.InsertNextId(3)
        
        faces.InsertNextId(4)
        faces.InsertNextId(1)
        faces.InsertNextId(2)
        faces.InsertNextId(5)
        faces.InsertNextId(4)
        
        faces.InsertNextId(4)
        faces.InsertNextId(2)
        faces.InsertNextId(0)
        faces.InsertNextId(3)
        faces.InsertNextId(5)
        
        ugrid = vtk.vtkUnstructuredGrid()
        ugrid.SetPoints(points)
        ugrid.InsertNextCell(vtk.VTK_POLYHEDRON, faces)
        
        geometry = vtk.vtkGeometryFilter()
        geometry.SetInputData(ugrid)
        return geometry
    
    def create_pyramid_source(self):
        """Create a pyramid (square pyramid)"""
        points = vtk.vtkPoints()
        points.InsertNextPoint(-1, -1, 0)  # Base: bottom-left
        points.InsertNextPoint(1, -1, 0)   # Base: bottom-right
        points.InsertNextPoint(1, 1, 0)    # Base: top-right
        points.InsertNextPoint(-1, 1, 0)   # Base: top-left
        points.InsertNextPoint(0, 0, 2)    # Apex
        
        # Create the pyramid cells
        pyramid = vtk.vtkPyramid()
        pyramid.GetPointIds().SetId(0, 0)
        pyramid.GetPointIds().SetId(1, 1)
        pyramid.GetPointIds().SetId(2, 2)
        pyramid.GetPointIds().SetId(3, 3)
        pyramid.GetPointIds().SetId(4, 4)
        
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(pyramid)
        
        ugrid = vtk.vtkUnstructuredGrid()
        ugrid.SetPoints(points)
        ugrid.InsertNextCell(pyramid.GetCellType(), pyramid.GetPointIds())
        
        # Convert to polydata for rendering
        geometry = vtk.vtkGeometryFilter()
        geometry.SetInputData(ugrid)
        return geometry
    
    def create_torus_source(self, major_radius, minor_radius):
        """Create a torus source (custom implementation)"""
        # This is a simplified torus implementation
        # For production, you might want to use vtkParametricTorus
        torus = vtk.vtkParametricTorus()
        torus.SetRingRadius(major_radius)
        torus.SetCrossSectionRadius(minor_radius)
        
        source = vtk.vtkParametricFunctionSource()
        source.SetParametricFunction(torus)
        source.SetUResolution(32)
        source.SetVResolution(16)
        
        return source
    
    def select_object(self, actor, multi_select=False):
        """Select an object - support multi-selection with Shift key"""
        if multi_select and actor in self.selected_actors:
            # Deselect if already selected (toggle)
            self.deselect_object(actor)
        elif multi_select:
            # Add to selection
            if actor not in self.selected_actors:
                self.selected_actors.append(actor)
                if actor in self.outline_actors:
                    self.outline_actors[actor].VisibilityOn()
                    # Ensure outline is at correct position
                    self.update_outline_position(actor)
        else:
            # Single selection - clear others
            self.deselect_all()
            self.selected_actors = [actor]
            if actor in self.outline_actors:
                self.outline_actors[actor].VisibilityOn()
                # Ensure outline is at correct position
                self.update_outline_position(actor)
        
        # Update gizmo based on current active tool - ONLY if we have an active tool
        if self.selected_actors and self.active_gizmo:
            print(f"Updating gizmo for active tool: {self.active_gizmo}")
            if self.active_gizmo == 'move':
                self.move_gizmo.show(self.selected_actors[0])
            elif self.active_gizmo == 'rotate':
                self.rotate_gizmo.show(self.selected_actors[0])
        else:
            print(f"No gizmo update - active_gizmo: {self.active_gizmo}, selected: {len(self.selected_actors)}")
        
        print(f"Selected {len(self.selected_actors)} objects")
    
    def deselect_object(self, actor):
        """Deselect a specific object"""
        if actor in self.selected_actors:
            self.selected_actors.remove(actor)
        if actor in self.outline_actors:
            self.outline_actors[actor].VisibilityOff()
    
    def deselect_all(self):
        """Deselect all objects"""
        # Hide ALL gizmos first
        self.move_gizmo.hide()
        self.rotate_gizmo.hide()
        self.scale_gizmo.hide()  # ADD THIS LINE
        
        # Then deselect objects
        for actor in self.selected_actors:
            if actor in self.outline_actors:
                self.outline_actors[actor].VisibilityOff()
        self.selected_actors.clear()
    
    def select_object(self, actor, multi_select=False):
        """Select an object - support multi-selection with Shift key"""
        if multi_select and actor in self.selected_actors:
            # Deselect if already selected (toggle)
            self.deselect_object(actor)
        elif multi_select:
            # Add to selection
            if actor not in self.selected_actors:
                self.selected_actors.append(actor)
                if actor in self.outline_actors:
                    self.outline_actors[actor].VisibilityOn()
                    # Ensure outline is at correct position
                    self.update_outline_position(actor)
        else:
            # Single selection - clear others
            self.deselect_all()  # This will hide gizmos
            self.selected_actors = [actor]
            if actor in self.outline_actors:
                self.outline_actors[actor].VisibilityOn()
                # Ensure outline is at correct position
                self.update_outline_position(actor)
        
        # Update gizmo based on current active tool - ONLY if we have an active tool AND selection
        if self.selected_actors and self.active_gizmo:
            print(f"Updating gizmo for active tool: {self.active_gizmo}")
            if self.active_gizmo == 'move':
                self.move_gizmo.show(self.selected_actors[0])
            elif self.active_gizmo == 'rotate':
                self.rotate_gizmo.show(self.selected_actors[0])
            elif self.active_gizmo == 'scale':
                self.scale_gizmo.show(self.selected_actors[0])
        else:
            print(f"No gizmo update - active_gizmo: {self.active_gizmo}, selected: {len(self.selected_actors)}")
        
        print(f"Selected {len(self.selected_actors)} objects")
    
    def create_outline(self, actor):
        """Create a box outline that properly follows rotation using vtkOutlineSource"""
        # Create a custom outline that matches the actor's bounds
        bounds = actor.GetBounds()
        
        # Create outline source
        outline_source = vtk.vtkOutlineSource()
        outline_source.SetBounds(bounds)
        
        outline_mapper = vtk.vtkPolyDataMapper()
        outline_mapper.SetInputConnection(outline_source.GetOutputPort())
        
        outline_actor = vtk.vtkActor()
        outline_actor.SetMapper(outline_mapper)
        outline_actor.GetProperty().SetColor(1.0, 0.8, 0.0)  # Yellow-orange highlight
        outline_actor.GetProperty().SetLineWidth(3.0)
        outline_actor.VisibilityOff()  # Start hidden
        
        # Use the same transform as the main actor
        outline_actor.SetUserTransform(actor.GetUserTransform())
        outline_actor.SetPosition(actor.GetPosition())
        outline_actor.SetOrientation(actor.GetOrientation())
        outline_actor.SetScale(actor.GetScale())
        
        self.renderer.AddActor(outline_actor)
        self.outline_actors[actor] = outline_actor
    
    def get_object_at_position(self, x, y):
        """Get the object at screen coordinates (x, y)"""
        picker = vtk.vtkPropPicker()
        picker.Pick(x, y, 0, self.renderer)
        
        picked_actor = picker.GetActor()
        if picked_actor and picked_actor in self.actors:
            return picked_actor
        return None
    
    def get_object_type(self, actor):
        """Get the type of object for an actor"""
        if actor in self.actors:
            index = self.actors.index(actor)
            if index < len(self.sources):
                source = self.sources[index]
                # Geometric Objects
                if isinstance(source, vtk.vtkSphereSource):
                    return "Sphere"
                elif isinstance(source, vtk.vtkCubeSource):
                    return "Cube"
                elif isinstance(source, vtk.vtkCylinderSource):
                    return "Cylinder"
                elif isinstance(source, vtk.vtkConeSource):
                    return "Cone"
                elif hasattr(self, 'create_pyramid_source') and hasattr(source, 'GetClassName') and 'Pyramid' in str(source.GetClassName()):
                    return "Pyramid"
                elif hasattr(source, 'GetParametricFunction') and source.GetParametricFunction():
                    param_func = source.GetParametricFunction()
                    if isinstance(param_func, vtk.vtkParametricTorus):
                        return "Torus"
                    elif isinstance(param_func, vtk.vtkParametricMobius):
                        return "Mobius Strip"
                    elif isinstance(param_func, vtk.vtkParametricKlein):
                        return "Klein Bottle"
                    elif isinstance(param_func, vtk.vtkParametricSuperToroid):
                        return "Super Toroid"
                    elif isinstance(param_func, vtk.vtkParametricSuperEllipsoid):
                        return "Super Ellipsoid"
                
                # Platonic Solids (Source Formats)
                elif isinstance(source, vtk.vtkPlatonicSolidSource):
                    solid_type = source.GetSolidType()
                    if solid_type == 0:  # Tetrahedron
                        return "Tetrahedron"
                    elif solid_type == 1:  # Cube (already handled above)
                        return "Cube"
                    elif solid_type == 2:  # Octahedron
                        return "Octahedron"
                    elif solid_type == 3:  # Icosahedron
                        return "Icosahedron"
                    elif solid_type == 4:  # Dodecahedron
                        return "Dodecahedron"
                
                # Cell Based Objects - check by structure
                elif hasattr(source, 'GetOutput') and source.GetOutput():
                    output = source.GetOutput()
                    if hasattr(output, 'GetCellType') and output.GetNumberOfCells() > 0:
                        cell_type = output.GetCellType(0)
                        if cell_type == vtk.VTK_VOXEL:
                            return "Voxel"
                        elif cell_type == vtk.VTK_HEXAHEDRON:
                            return "Hexahedron"
                        elif cell_type == vtk.VTK_POLYHEDRON:
                            return "Polyhedron"
                        elif cell_type == vtk.VTK_CONVEX_POINT_SET:
                            return "Convex Point Set"
                
                # Check for convex point set by points count and structure
                elif hasattr(source, 'GetOutput') and source.GetOutput():
                    output = source.GetOutput()
                    if output.GetNumberOfPoints() > 10:  # Convex sets usually have multiple points
                        points = output.GetPoints()
                        if points and points.GetNumberOfPoints() >= 8:  # Reasonable minimum for convex set
                            # Check if it's likely a convex hull
                            bounds = output.GetBounds()
                            if bounds and abs(bounds[1] - bounds[0]) > 0:
                                return "Convex Point Set"
                
                # Isosurface Objects - check by name or characteristics
                elif hasattr(source, 'GetOutput') and source.GetOutput():
                    output = source.GetOutput()
                    # Check for gyroid-like structures (complex periodic surfaces)
                    if output.GetNumberOfPoints() > 1000:  # Isosurfaces are usually dense
                        bounds = output.GetBounds()
                        if bounds and abs(bounds[1] - bounds[0]) > 4:  # Larger bounds typical of isosurfaces
                            polydata = source.GetOutput()
                            if polydata and polydata.GetNumberOfPolys() > 500:
                                # Try to identify by point distribution
                                points = polydata.GetPoints()
                                if points:
                                    # Sample some points to detect patterns
                                    import numpy as np
                                    sample_points = []
                                    for i in range(min(10, points.GetNumberOfPoints())):
                                        sample_points.append(points.GetPoint(i))
                                    
                                    # Very basic pattern detection
                                    x_coords = [p[0] for p in sample_points]
                                    y_coords = [p[1] for p in sample_points]
                                    z_coords = [p[2] for p in sample_points]
                                    
                                    # Check for periodic patterns
                                    x_range = max(x_coords) - min(x_coords) if x_coords else 0
                                    y_range = max(y_coords) - min(y_coords) if y_coords else 0
                                    z_range = max(z_coords) - min(z_coords) if z_coords else 0
                                    
                                    if x_range > 3 and y_range > 3 and z_range > 3:
                                        return "Isosurface"
                
                # Check for specific isosurface types by source characteristics
                elif hasattr(source, 'GetClassName'):
                    class_name = str(source.GetClassName())
                    if 'Contour' in class_name:
                        return "Isosurface"
            
            # Fallback: Check if we stored object type during creation
            if hasattr(actor, '_object_type'):
                return getattr(actor, '_object_type', "Object")
        
        return "Object"
    
    def update_outline_position(self, actor):
        """Update outline position AND orientation to follow the actor"""
        if actor in self.outline_actors:
            outline_actor = self.outline_actors[actor]
            outline_actor.SetPosition(actor.GetPosition())
            outline_actor.SetOrientation(actor.GetOrientation())
            
    def update_all_outlines(self):
        """Update all outlines to follow their respective actors"""
        for actor, outline_actor in self.outline_actors.items():
            outline_actor.SetPosition(actor.GetPosition())
            outline_actor.SetOrientation(actor.GetOrientation())
        
    def change_color(self, color):
        """Change the color of all objects"""
        print(f"Changing color to {color}")
        self.current_color = color
        
        # Update color of all actors
        for actor in self.actors:
            actor.GetProperty().SetColor(color)
            
    def move_object(self, actor, delta_x, delta_y, delta_z):
        """Move an object and update its outline (and lights if needed)"""
        current_pos = actor.GetPosition()
        new_pos = [
            current_pos[0] + delta_x,
            current_pos[1] + delta_y, 
            current_pos[2] + delta_z
        ]
        
        actor.SetPosition(new_pos)
        self.update_outline_position(actor)

        # NEW: if this actor represents a light icon, sync its vtkLight
        if hasattr(self, "light_manager"):
            self.light_manager.sync_light_for_actor(actor)
    
    def clear_objects(self):
        """Remove all objects from the scene"""
        # Remove outline actors first
        for outline_actor in self.outline_actors.values():
            self.renderer.RemoveActor(outline_actor)
        self.outline_actors.clear()
        
        # Remove main actors
        for actor in self.actors:
            self.renderer.RemoveActor(actor)
        self.actors.clear()
        self.mappers.clear()
        self.sources.clear()
        self.selected_actors.clear()  # FIXED: Clear the list, not single attribute
        
    def set_active_tool(self, tool_name):
        """Set the active tool and show/hide appropriate gizmo"""
        print(f"ObjectManager.set_active_tool: {tool_name}, selected_actors: {len(self.selected_actors)}")
        self.active_gizmo = tool_name
        
        # Hide all gizmos first
        self.move_gizmo.hide()
        self.rotate_gizmo.hide()
        self.scale_gizmo.hide()  # ADD THIS LINE
        
        # Show appropriate gizmo ONLY if we have a selection
        if self.selected_actors:
            if tool_name == 'move':
                print("Showing move gizmo")
                self.move_gizmo.show(self.selected_actors[0])
            elif tool_name == 'rotate':
                print("Showing rotate gizmo")
                self.rotate_gizmo.show(self.selected_actors[0])
            elif tool_name == 'scale':  # ADD SCALE CASE
                print("Showing scale gizmo")
                self.scale_gizmo.show(self.selected_actors[0])
            else:
                print(f"Tool {tool_name} doesn't use gizmo")
        else:
            print("No selected actors - keeping gizmos hidden")
            
    def create_camera(self):
        """Create a camera object in the scene"""
        print("Creating camera...")
        
        # Create camera object with default position
        camera_obj = CameraObject()
        
        # Add camera actor to the scene
        self.renderer.AddActor(camera_obj.actor)
        self.actors.append(camera_obj.actor)
        
        # Store camera object reference
        camera_obj.actor._is_camera = True
        camera_obj.actor._camera_object = camera_obj
        camera_obj.actor._object_type = "Camera"
        self.cameras.append(camera_obj)
        
        # Create outline and select the camera
        self.create_outline(camera_obj.actor)
        self.select_object(camera_obj.actor)
        
        print("Camera created and selected")
        return camera_obj.actor
    
    def move_object(self, actor, delta_x, delta_y, delta_z):
        """Move an object and update its outline (and lights if needed)"""
        current_pos = actor.GetPosition()
        new_pos = [
            current_pos[0] + delta_x,
            current_pos[1] + delta_y, 
            current_pos[2] + delta_z
        ]
        
        actor.SetPosition(new_pos)
        self.update_outline_position(actor)

        # NEW: if this actor is a camera, update its camera properties
        if hasattr(actor, '_is_camera') and actor._is_camera:
            camera_obj = actor._camera_object
            camera_obj.set_position(new_pos)
            # Update focal point to maintain view direction
            direction = camera_obj.get_view_direction()
            new_focal = [
                new_pos[0] + direction[0] * 10,
                new_pos[1] + direction[1] * 10,
                new_pos[2] + direction[2] * 10
            ]
            camera_obj.set_focal_point(new_focal)

        # NEW: if this actor represents a light icon, sync its vtkLight
        if hasattr(self, "light_manager"):
            self.light_manager.sync_light_for_actor(actor)
        
class BlenderLikeGrid:
    def __init__(self):
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.1, 0.1, 0.1)  # Dark background
        
        self.create_grid()
        self.create_axes()
        
    def create_grid(self):
        grid_size = 200
        spacing = 1.0
        
        # Create main grid assembly
        self.grid_assembly = vtk.vtkAssembly()
        
        for i in range(-grid_size, grid_size + 1):
            distance = abs(i)
            
            if i % 10 == 0:
                base_opacity = 0.6
                line_width = 1.5
            else:
                base_opacity = 0.3
                line_width = 1.0
            
            fade_factor = max(0, 1 - (distance / grid_size) * 1.5)
            opacity = base_opacity * fade_factor
            effective_line_width = line_width * fade_factor

            # X-direction lines
            line_x = self.create_line(
                [i * spacing, -grid_size * spacing, 0],
                [i * spacing, grid_size * spacing, 0],
                [1.0, 1.0, 1.0], opacity, effective_line_width
            )
            self.grid_assembly.AddPart(line_x)
            
            # Y-direction lines
            line_y = self.create_line(
                [-grid_size * spacing, i * spacing, 0],
                [grid_size * spacing, i * spacing, 0],
                [1.0, 1.0, 1.0], opacity, effective_line_width
            )
            self.grid_assembly.AddPart(line_y)
        
        self.renderer.AddActor(self.grid_assembly)
    
    def create_line(self, point1, point2, color, opacity, line_width=1.0):
        line_source = vtk.vtkLineSource()
        line_source.SetPoint1(point1)
        line_source.SetPoint2(point2)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(line_source.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetLineWidth(line_width)
        
        return actor
    
    def create_axes(self):
        # X-axis (Red)
        x_axis = self.create_line([-200, 0, 0], [200, 0, 0], [1.0, 0.0, 0.0], 1.0, 2.0)
        
        # Y-axis (Green)
        y_axis = self.create_line([0, -200, 0], [0, 200, 0], [0.0, 1.0, 0.0], 1.0, 2.0)
        
        # Z-axis (Blue)
        z_axis = self.create_line([0, 0, -200], [0, 0, 200], [0.0, 0.0, 1.0], 1.0, 2.0)
        
        self.renderer.AddActor(x_axis)
        self.renderer.AddActor(y_axis)
        self.renderer.AddActor(z_axis)
    
    def get_renderer(self):
        return self.renderer
    
class LightManager:
    """
    Manages Blender-style lights:
    - Tangible light icons (vtkActor) that you can move with gizmos
    - Underlying vtkLight objects attached to the renderer
    """

    def __init__(self, renderer, object_manager):
        self.renderer = renderer
        self.object_manager = object_manager

        # actor -> info dict
        # info = {
        #   'type': 'point' | 'sun' | 'spot' | 'area' | 'mesh' | 'world',
        #   'vtk_light': vtkLight (or list for area),
        #   'extra': {...}
        # }
        self.light_objects = {}
        self.world_light = None
        self.object_manager.light_manager = self

    # --------- PUBLIC API ---------

    def create_light(self, light_type: str):
        """Entry point used by UI."""
        light_type = light_type.lower()
        if light_type == "point":
            return self._create_point_light()
        elif light_type == "sun":
            return self._create_sun_light()
        elif light_type == "spot":
            return self._create_spot_light()
        elif light_type == "area":
            return self._create_area_light()
        elif light_type == "mesh":
            return self._create_mesh_light()
        elif light_type == "world":
            return self._create_world_light()
        else:
            print(f"Unknown light type: {light_type}")
            return None

    def sync_light_for_actor(self, actor):
        """
        Called whenever an actor moves / rotates.
        If that actor is a light icon, update its vtkLight.
        """
        if actor not in self.light_objects:
            return

        info = self.light_objects[actor]
        light_type = info['type']

        if light_type == "point":
            self._update_point_light(actor, info)
        elif light_type == "sun":
            self._update_sun_light(actor, info)
        elif light_type == "spot":
            self._update_spot_light(actor, info)
        elif light_type == "area":
            self._update_area_light(actor, info)
        elif light_type == "mesh":
            self._update_mesh_light(actor, info)
        # world light has no icon in scene (global)

    # --------- INTERNAL HELPERS ---------

    def _add_icon_actor(self, source, color, position=(0, 5, 0)):
        """Utility to create a small light icon actor and register with ObjectManager."""
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetAmbient(1.0)    # make it 'glow' a bit
        actor.GetProperty().SetDiffuse(0.2)
        actor.GetProperty().SetSpecular(0.0)
        actor.GetProperty().LightingOff()
        actor.SetScale(1.5, 1.5, 1.5)
        actor.SetPosition(*position)

        # Mark as special type in scene list
        actor._object_type = "Light"

        # Add via ObjectManager so it gets outline & gizmos
        self.object_manager.add_external_model(actor, mapper, source)

        return actor

    # --------- POINT LIGHT ---------

    def _create_point_light(self):
        # Small glowing sphere
        src = vtk.vtkSphereSource()
        src.SetRadius(3.0)
        src.SetPhiResolution(16)
        src.SetThetaResolution(16)

        actor = self._add_icon_actor(src, (1.0, 0.9, 0.3), position=(0, 5, 0))

        light = vtk.vtkLight()
        light.SetLightTypeToSceneLight()
        light.SetPositional(True)
        light.SetColor(1.0, 1.0, 1.0)
        light.SetIntensity(1.2)

        self.renderer.AddLight(light)

        info = {
            "type": "point",
            "vtk_light": light
        }
        self.light_objects[actor] = info
        self._update_point_light(actor, info)
        print("Created Point Light")
        return actor

    def _update_point_light(self, actor, info):
        light = info["vtk_light"]
        pos = actor.GetPosition()
        light.SetPosition(*pos)
        # Aim at origin by default
        light.SetFocalPoint(0.0, 0.0, 0.0)

    # --------- SUN (DIRECTIONAL) LIGHT ---------

    def _create_sun_light(self):
        # Simple arrow icon pointing toward origin (position encodes direction)
        arrow = vtk.vtkArrowSource()
        arrow.SetTipLength(0.4)
        arrow.SetTipRadius(0.15)
        arrow.SetShaftRadius(0.05)
        arrow.SetResolution(24)

        actor = self._add_icon_actor(arrow, (1.0, 1.0, 0.0), position=(0, 10, 0))
        actor.SetScale(3.0, 3.0, 3.0)
        actor._object_type = "Sun Light"

        light = vtk.vtkLight()
        light.SetLightTypeToSceneLight()
        light.SetPositional(False)  # directional
        light.SetColor(1.0, 1.0, 0.9)
        light.SetIntensity(1.0)

        self.renderer.AddLight(light)

        info = {
            "type": "sun",
            "vtk_light": light
        }
        self.light_objects[actor] = info
        self._update_sun_light(actor, info)
        print("Created Sun (directional) Light")
        return actor

    def _update_sun_light(self, actor, info):
        light = info["vtk_light"]
        pos = actor.GetPosition()

        # Treat vector from origin to icon as sun direction
        # (moving the icon around sphere changes direction)
        direction = [-pos[0], -pos[1], -pos[2]]
        if direction == [0.0, 0.0, 0.0]:
            direction = [0.0, -1.0, -0.5]

        # position doesn't really matter for directional, but set for consistency
        light.SetPosition(*pos)
        light.SetFocalPoint(
            pos[0] + direction[0],
            pos[1] + direction[1],
            pos[2] + direction[2],
        )

    # --------- SPOT LIGHT ---------

    def _create_spot_light(self):
        # Cone with base at icon position, pointing to origin
        cone = vtk.vtkConeSource()
        cone.SetHeight(4.0)
        cone.SetRadius(1.2)
        cone.SetResolution(24)

        actor = self._add_icon_actor(cone, (0.8, 0.9, 1.0), position=(3, 6, 3))
        actor._object_type = "Spot Light"

        light = vtk.vtkLight()
        light.SetLightTypeToSceneLight()
        light.SetPositional(True)
        light.SetColor(1.0, 1.0, 0.9)
        light.SetConeAngle(30.0)
        light.SetExponent(10.0)   # falloff inside cone
        light.SetIntensity(1.3)

        self.renderer.AddLight(light)

        info = {
            "type": "spot",
            "vtk_light": light
        }
        self.light_objects[actor] = info
        self._update_spot_light(actor, info)
        print("Created Spot Light")
        return actor

    def _update_spot_light(self, actor, info):
        light = info["vtk_light"]
        pos = actor.GetPosition()

        # Aim at origin for now (simple version)
        target = (0.0, 0.0, 0.0)
        light.SetPosition(*pos)
        light.SetFocalPoint(*target)

    # --------- AREA LIGHT (APPROX: CLUSTER OF POINTS) ---------

    def _create_area_light(self):
        """
        Simple approximation: 4 point lights in a square,
        plus a thin rectangle icon.
        """
        plane = vtk.vtkPlaneSource()
        plane.SetOrigin(-2.0, -1.0, 0)
        plane.SetPoint1( 2.0, -1.0, 0)
        plane.SetPoint2(-2.0,  1.0, 0)
        plane.SetResolution(1, 1)

        actor = self._add_icon_actor(plane, (0.7, 1.0, 0.7), position=(-5, 5, 0))
        actor._object_type = "Area Light"

        # Create 4 small point lights around center
        lights = []
        offsets = [
            (-0.5, -0.5, 0),
            (0.5, -0.5, 0),
            (-0.5, 0.5, 0),
            (0.5, 0.5, 0),
        ]
        for _ in offsets:
            l = vtk.vtkLight()
            l.SetLightTypeToSceneLight()
            l.SetPositional(True)
            l.SetColor(1.0, 1.0, 1.0)
            l.SetIntensity(0.4)  # 4 * 0.4 = 1.6 total approx
            self.renderer.AddLight(l)
            lights.append(l)

        info = {
            "type": "area",
            "vtk_light": lights,
            "offsets": offsets
        }
        self.light_objects[actor] = info
        self._update_area_light(actor, info)
        print("Created Area Light (approx via 4 points)")
        return actor

    def _update_area_light(self, actor, info):
        lights = info["vtk_light"]
        offsets = info["offsets"]
        pos = actor.GetPosition()

        for o, l in zip(offsets, lights):
            l.SetPosition(pos[0] + o[0], pos[1] + o[1], pos[2] + o[2])
            l.SetFocalPoint(0.0, 0.0, 0.0)

    # --------- MESH LIGHT (SIMPLE VERSION) ---------

    def _create_mesh_light(self):
        """
        Simple version: use currently selected mesh as a light:
        - Boost its ambient/diffuse
        - Add a point light at its center
        """
        if not self.object_manager.selected_actors:
            print("Mesh Light: no mesh selected.")
            return None

        target_actor = self.object_manager.selected_actors[0]
        bounds = target_actor.GetBounds()
        center = (
            (bounds[0] + bounds[1]) / 2.0,
            (bounds[2] + bounds[3]) / 2.0,
            (bounds[4] + bounds[5]) / 2.0,
        )

        # Boost material
        prop = target_actor.GetProperty()
        prop.SetAmbient(0.8)
        prop.SetDiffuse(0.8)
        prop.SetEmission(0.6)

        # Add helper light at centroid
        light = vtk.vtkLight()
        light.SetLightTypeToSceneLight()
        light.SetPositional(True)
        light.SetColor(1.0, 0.9, 0.7)
        light.SetIntensity(1.0)
        light.SetPosition(*center)
        light.SetFocalPoint(0.0, 0.0, 0.0)

        self.renderer.AddLight(light)

        info = {
            "type": "mesh",
            "vtk_light": light,
            "target_actor": target_actor
        }
        # We don't create an extra icon actor; the mesh itself is the "light"
        self.light_objects[target_actor] = info
        print("Created Mesh Light on selected object")
        return target_actor

    def _update_mesh_light(self, actor, info):
        light = info["vtk_light"]
        bounds = actor.GetBounds()
        center = (
            (bounds[0] + bounds[1]) / 2.0,
            (bounds[2] + bounds[3]) / 2.0,
            (bounds[4] + bounds[5]) / 2.0,
        )
        light.SetPosition(*center)
        
    def set_intensity_for_actor(self, actor, intensity: float):
        """
        Set light intensity for the light represented by this icon actor.
        `intensity` is a scalar (e.g. 0.0–2.0).
        """
        info = self.light_objects.get(actor)
        if not info:
            return
        
        light_type = info["type"]
        
        # Clamp intensity
        intensity = max(0.0, float(intensity))
        
        if light_type in ("point", "spot", "sun", "mesh", "world"):
            vtk_light = info.get("vtk_light")
            if vtk_light:
                vtk_light.SetIntensity(intensity)
        
        elif light_type == "area":
            # area lights are stored as list of 4 vtkLight in info["vtk_lights"]
            lights = info.get("vtk_lights", [])
            if lights:
                # Distribute intensity over all 4 so total brightness feels consistent
                per_light = intensity / max(1, len(lights))
                for l in lights:
                    l.SetIntensity(per_light)
        
        # Optionally re-render
        self.renderer.GetRenderWindow().Render()

    # --------- WORLD LIGHT / ENVIRONMENT ---------

    def _create_world_light(self):
        """
        World lighting: set a soft headlight + a bit of global ambient.
        No tangible icon (matches Blender's 'World' concept).
        """
        if self.world_light is None:
            wl = vtk.vtkLight()
            wl.SetLightTypeToHeadlight()
            wl.SetColor(0.9, 0.9, 1.0)
            wl.SetIntensity(0.7)
            self.renderer.AddLight(wl)
            self.world_light = wl

            # Slight ambient for renderer
            self.renderer.SetAmbient(0.2, 0.2, 0.25)
            print("World lighting enabled")
        else:
            # Toggle off for now
            self.renderer.RemoveLight(self.world_light)
            self.world_light = None
            self.renderer.SetAmbient(0.0, 0.0, 0.0)
            print("World lighting disabled")

        return None

class VTKWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create VTK renderer and grid
        self.grid = BlenderLikeGrid()
        self.renderer = self.grid.get_renderer()
        
        # Create object manager
        self.object_manager = ObjectManager(self.renderer)

        # Create light manager (Blender-like lights)
        self.light_manager = LightManager(self.renderer, self.object_manager)
        
        # Create measurement tool - ADD THIS
        self.measurement_tool = MeasurementTool(self.renderer)
        
        # Create VTK widget
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.render_window = self.vtk_widget.GetRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        
        # Get the interactor
        self.interactor = self.render_window.GetInteractor()
        
        self.original_camera = None
        self.backup_camera = None
        self.is_camera_view_active = False
        self.is_camera_view_mode = False
        self.active_camera_object = None
        
        # Camera setup
        self.camera = self.renderer.GetActiveCamera()
        
        # Camera state - spherical coordinates
        self.camera_radius = 15.0
        self.camera_theta = math.radians(45.0)  # Vertical angle in radians
        self.camera_phi = math.radians(45.0)    # Horizontal angle in radians
        
        # Blender-like rotation properties
        self.last_mouse_pos = QPoint()
        self.is_rotating_view = False  # For camera/view rotation
        self.is_rotating_object = False  # For object rotation  # ADD THIS LINE
        
        # Rotation pivot (like Blender's 3D cursor concept)
        self.rotation_pivot = [0, 0, 0]  # Focus on origin by default
        
        self.setup_layout()
        self.update_camera_position()
        
        # Tool states
        self.current_tool = 'select'
        self.box_select_rubberband = BoxSelectRubberBand(self.vtk_widget)  # Parent to vtk_widget instead of self
        self.box_select_start = QPoint()
        self.is_box_selecting = False
        self.main_window = None
        
        # Move tool state
        self.is_moving = False
        self.move_axis = None
        self.move_start_point = QPoint()
        
        # --- NEW: free-move drag state (screen-plane dragging) ---
        self.free_move_mouse_world_start = None  # world position under mouse when drag starts
        self.free_move_actor_start = None        # original actor position when drag starts
        
        # Rotate tool state
        self.rotate_axis = None 
        self.rotate_start_point = QPoint()  
        
        self.is_scaling = False
        self.scale_axis = None
        self.scale_start_point = QPoint()
        self.original_scale = [1.0, 1.0, 1.0]
        
        # Measurement tool state - ADD THIS
        self.is_measuring = False
        self.measure_start_point = QPoint()
        
        # COMPLETELY disable VTK interactions
        self.disable_vtk_interactions()
        
        # Set up cursor
        self.setCursor(Qt.OpenHandCursor)  # Blender-like open hand cursor
        
        # Start the interactor
        self.interactor.Initialize()
        
    def reset_to_main_view(self):
        """Reset to the main camera view from camera view mode"""
        if self.is_camera_view_mode and self.backup_camera:
            # Restore the backup camera
            self.renderer.SetActiveCamera(self.backup_camera)
            self.camera = self.backup_camera
            self.is_camera_view_mode = False
            self.active_camera_object = None
            
            # Update the camera position for the main view
            self.update_camera_position()
            
            print("Returned to main view")
        else:
            # Fallback to regular reset
            self.reset_view()
    
    def set_camera_view(self, camera_obj):
        """Set the view to a specific camera's perspective"""
        if not camera_obj or not hasattr(camera_obj, 'vtk_camera'):
            return False
            
        # Backup the current camera if this is the first time switching to camera view
        if not self.is_camera_view_mode:
            self.backup_camera = vtk.vtkCamera()
            self.backup_camera.DeepCopy(self.camera)
            self.is_camera_view_mode = True
        
        # Update the camera object to match its current actor position
        camera_obj.set_position(camera_obj.actor.GetPosition())
        
        # Calculate focal point based on camera orientation
        camera_direction = camera_obj.get_view_direction()
        focal_point = [
            camera_obj.position[0] + camera_direction[0] * 10,
            camera_obj.position[1] + camera_direction[1] * 10,
            camera_obj.position[2] + camera_direction[2] * 10
        ]
        camera_obj.set_focal_point(focal_point)
        
        # Set the renderer to use the camera object's camera
        self.renderer.SetActiveCamera(camera_obj.vtk_camera)
        self.camera = camera_obj.vtk_camera
        self.active_camera_object = camera_obj
        
        # Update the renderer
        self.renderer.ResetCameraClippingRange()
        self.render_window.Render()
        
        print(f"Switched to camera view: {camera_obj.name}")
        return True

    def save_view_image(self, file_path):
        """Save the current view as an image file"""
        try:
            # Create window to image filter
            window_to_image_filter = vtk.vtkWindowToImageFilter()
            window_to_image_filter.SetInput(self.render_window)
            window_to_image_filter.SetScale(1)  # Image quality
            window_to_image_filter.SetInputBufferTypeToRGB()
            window_to_image_filter.ReadFrontBufferOff()  # Read from the back buffer
            window_to_image_filter.Update()

            # Create writer based on file extension
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext in ['.jpg', '.jpeg']:
                writer = vtk.vtkJPEGWriter()
            elif file_ext in ['.png']:
                writer = vtk.vtkPNGWriter()
            elif file_ext in ['.bmp']:
                writer = vtk.vtkBMPWriter()
            elif file_ext in ['.tiff', '.tif']:
                writer = vtk.vtkTIFFWriter()
            else:
                # Default to PNG
                file_path += '.png'
                writer = vtk.vtkPNGWriter()

            writer.SetFileName(file_path)
            writer.SetInputConnection(window_to_image_filter.GetOutputPort())
            writer.Write()
            
            print(f"View saved as: {file_path}")
            return True
            
        except Exception as e:
            print(f"Error saving view image: {e}")
            return False
        
    def disable_vtk_interactions(self):
        """Completely disable VTK's default interactions"""
        # Create a minimal interactor style that does nothing
        class EmptyInteractorStyle(vtk.vtkInteractorStyle):
            def __init__(self):
                super().__init__()
                
            def OnLeftButtonDown(self):
                pass
                
            def OnMiddleButtonDown(self):
                pass
                
            def OnRightButtonDown(self):
                pass
                
            def OnMouseWheelForward(self):
                pass
                
            def OnMouseWheelBackward(self):
                pass
                
            def OnMouseMove(self):
                pass
        
        self.interactor_style = EmptyInteractorStyle()
        self.interactor.SetInteractorStyle(self.interactor_style)
        
        # Also disable the VTK widget's event processing
        self.vtk_widget.setEnabled(False)
        
    def setup_layout(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.vtk_widget)
        
    def get_world_position_from_mouse(self, mouse_pos):
        """Convert mouse position to world coordinates - UNIVERSAL VERSION"""
        # Delegate to measurement tool for universal picking
        if hasattr(self, 'measurement_tool') and self.measurement_tool.is_active:
            return self.measurement_tool.get_world_position_from_mouse(mouse_pos.x(), mouse_pos.y())
        else:
            # For other tools, use object-only picking
            renderer_size = self.renderer.GetSize()
            if renderer_size[0] == 0 or renderer_size[1] == 0:
                return None
                
            # Convert to VTK coordinates
            x = mouse_pos.x()
            y = renderer_size[1] - mouse_pos.y()
            
            # Use picker to get world position (object-only)
            picker = vtk.vtkPropPicker()
            picker.Pick(x, y, 0, self.renderer)
            pick_pos = picker.GetPickPosition()
            return pick_pos if pick_pos != [0, 0, 0] else None
        
    def update_camera_position(self):
        """Update camera position using proper spherical coordinates"""
        # Convert spherical coordinates to Cartesian
        x = self.camera_radius * math.sin(self.camera_theta) * math.cos(self.camera_phi)
        y = self.camera_radius * math.sin(self.camera_theta) * math.sin(self.camera_phi)
        z = self.camera_radius * math.cos(self.camera_theta)
        
        # Add pivot offset (orbit around the pivot point)
        camera_pos = [
            x + self.rotation_pivot[0],
            y + self.rotation_pivot[1], 
            z + self.rotation_pivot[2]
        ]
        
        # Set camera position (orbiting around pivot)
        self.camera.SetPosition(camera_pos)
        
        # Always look at the pivot point
        self.camera.SetFocalPoint(self.rotation_pivot)
        
        # Calculate proper up vector to maintain orientation
        if abs(self.camera_theta) < 0.01 or abs(self.camera_theta - math.pi) < 0.01:
            self.camera.SetViewUp(0, 1, 0)
        else:
            self.camera.SetViewUp(0, 0, 1)
        
        self.renderer.ResetCameraClippingRange()
        self.render_window.Render()
        
    def load_external_model(self, file_path, file_type):
        """Load an external 3D model file"""
        try:
            reader = None
            mapper = vtk.vtkPolyDataMapper()
            
            if file_type.lower() == 'obj':
                reader = vtk.vtkOBJReader()
                reader.SetFileName(file_path)
                mapper.SetInputConnection(reader.GetOutputPort())
                
            elif file_type.lower() == 'ply':
                reader = vtk.vtkPLYReader()
                reader.SetFileName(file_path)
                mapper.SetInputConnection(reader.GetOutputPort())
                
            elif file_type.lower() == 'stl':
                reader = vtk.vtkSTLReader()
                reader.SetFileName(file_path)
                mapper.SetInputConnection(reader.GetOutputPort())
                
            elif file_type.lower() == '3ds':
                # For 3DS files, we might need vtk3DSReader or alternative approach
                try:
                    reader = vtk.vtk3DSReader()
                    reader.SetFileName(file_path)
                    # 3DS files can contain multiple objects, so we need to handle them
                    reader.Update()
                    
                    # Get the output which might be a vtkPolyData or assembly
                    output = reader.GetOutput()
                    if output:
                        if isinstance(output, vtk.vtkPolyData):
                            mapper.SetInputData(output)
                        else:
                            # For assemblies, extract the first polydata
                            geometry_filter = vtk.vtkGeometryFilter()
                            geometry_filter.SetInputData(output)
                            mapper.SetInputConnection(geometry_filter.GetOutputPort())
                    else:
                        raise Exception("No data in 3DS file")
                        
                except Exception as e:
                    print(f"3DS reader failed: {e}")
                    # Fallback: try using generic importer
                    return self.load_model_with_importer(file_path, '3ds')
            
            else:
                print(f"Unsupported file type: {file_type}")
                return False
            
            # Check if reader successfully read the file
            if reader:
                reader.Update()
                if reader.GetOutput() and reader.GetOutput().GetNumberOfPoints() > 0:
                    return self.create_actor_from_mapper(mapper, file_path)
                else:
                    print(f"No data found in file: {file_path}")
                    return False
            else:
                return False
                
        except Exception as e:
            print(f"Error loading {file_type} file {file_path}: {e}")
            return False
    
    def load_model_with_importer(self, file_path, file_type):
        """Alternative method using vtkImporter for complex formats"""
        try:
            if file_type.lower() == '3ds':
                # Try using VRML importer as alternative for 3DS
                importer = vtk.vtkVRMLImporter()
                importer.SetFileName(file_path)
                importer.Read()
                importer.Update()
                
                # Get the actors from the renderer
                actors = importer.GetRenderer().GetActors()
                actors.InitTraversal()
                actor = actors.GetNextActor()
                
                if actor:
                    # Clone the actor and add to our scene
                    new_actor = vtk.vtkActor()
                    new_actor.SetMapper(actor.GetMapper())
                    new_actor.SetPosition(0, 0, 0)
                    new_actor.GetProperty().SetColor(1.0, 1.0, 1.0)  # Default white color
                    
                    self.object_manager.actors.append(new_actor)
                    self.renderer.AddActor(new_actor)
                    
                    # Create outline and select the new object
                    self.object_manager.create_outline(new_actor)
                    self.object_manager.select_object(new_actor)
                    
                    self.update_camera_position()
                    return True
                else:
                    return False
            else:
                return False
                
        except Exception as e:
            print(f"Importer method failed for {file_path}: {e}")
            return False
    
    def create_actor_from_mapper(self, mapper, file_path):
        """Create an actor from mapper and add to scene"""
        try:
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.SetPosition(0, 0, 0)
            actor.GetProperty().SetColor(1.0, 1.0, 1.0)  # Default white color
            
            # Add to object manager
            self.object_manager.actors.append(actor)
            self.object_manager.mappers.append(mapper)
            
            # Add to renderer
            self.renderer.AddActor(actor)
            
            # Create outline and select the new object
            self.object_manager.create_outline(actor)
            self.object_manager.select_object(actor)
            
            # Auto-fit camera to the new object
            self.renderer.ResetCamera()
            self.update_camera_position()
            
            print(f"Successfully loaded model: {file_path}")
            return True
            
        except Exception as e:
            print(f"Error creating actor: {e}")
            return False
        
    def on_tool_changed(self, tool_name):
        """Handle tool changes from the toolbar"""
        self.current_tool = tool_name
        print(f"Active tool: {tool_name}")
        
        # ALWAYS hide both gizmos first, regardless of selection
        self.object_manager.move_gizmo.hide()
        self.object_manager.rotate_gizmo.hide()
        
        # Update cursor and show appropriate gizmo
        if tool_name == 'select':
            self.setCursor(Qt.ArrowCursor)
            # Gizmos already hidden above
        elif tool_name == 'move':
            self.setCursor(Qt.SizeAllCursor)
            if self.object_manager.selected_actors:
                print("Calling move gizmo show from on_tool_changed")
                self.object_manager.move_gizmo.show(self.object_manager.selected_actors[0])
            else:
                print("No selected object for move gizmo")
        elif tool_name == 'rotate':
            self.setCursor(Qt.CrossCursor)
            if self.object_manager.selected_actors:
                print("Calling rotate gizmo show from on_tool_changed")
                self.object_manager.rotate_gizmo.show(self.object_manager.selected_actors[0])
            else:
                print("No selected object for rotate gizmo")
        elif tool_name == 'scale':  # ADD SCALE CASE
            self.setCursor(Qt.SizeAllCursor)
            if self.object_manager.selected_actors:
                self.object_manager.scale_gizmo.show(self.object_manager.selected_actors[0])
        elif tool_name == 'box_select':
            self.setCursor(Qt.CrossCursor)
            # Gizmos already hidden above
        
        # CRITICAL: Update the object manager's active tool
        self.object_manager.set_active_tool(tool_name)
        
        # Debug info
        self.debug_gizmo_status()
        
        # Force a render to update the display
        self.render_window.Render()
        
    def mousePressEvent(self, event: QMouseEvent):
        self.vtk_widget.clearFocus()
        self.setFocus()
        
        if event.button() == Qt.LeftButton:
            if self.current_tool == 'select':
                multi_select = bool(event.modifiers() & Qt.ShiftModifier)
                self.handle_object_selection(event.pos(), multi_select)
            elif self.current_tool == 'move':
                self.handle_move_start(event.pos())
            elif self.current_tool == 'rotate': 
                self.handle_rotate_start(event.pos())
            elif self.current_tool == 'scale':  
                self.handle_scale_start(event.pos())
            elif self.current_tool == 'box_select':
                self.start_box_selection(event.pos())
            elif self.current_tool == 'measure':  # UNIVERSAL MEASUREMENT TOOL
                # Directly call measurement tool with screen coordinates
                if self.measurement_tool.handle_click(event.pos().x(), event.pos().y()):
                    self.render_window.Render()
                else:
                    # Fall back to object selection if measurement didn't handle it
                    self.handle_object_selection(event.pos(), False)
            else:
                self.handle_object_selection(event.pos(), False)
            event.accept()
            
        elif event.button() == Qt.MiddleButton:
            self.is_rotating_view = True  # Camera/view rotation
            self.last_mouse_pos = event.pos()
            
            if event.modifiers() & Qt.ShiftModifier:
                self.setCursor(Qt.SizeAllCursor)
            else:
                self.setCursor(Qt.ClosedHandCursor)
            
            event.accept()
        else:
            super().mousePressEvent(event)
            
    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self.is_scaling:
            self.is_scaling = False
            self.scale_axis = None
            
            # RESUME property panel updates after scaling
            if hasattr(self, 'main_window') and self.main_window:
                self.main_window.right_panel.transform_widget.resume_updates()
                # Force one update to sync the final values
                self.main_window.right_panel.transform_widget.update_from_selection()
            
            print("Finished scaling object")
            event.accept()
        elif event.button() == Qt.LeftButton and self.is_moving:
            self.is_moving = False
            self.move_axis = None
            # reset free-move state
            self.free_move_mouse_world_start = None
            self.free_move_actor_start = None
            print("Finished moving object")
            event.accept()
        elif event.button() == Qt.LeftButton and self.is_rotating_object:  # Fixed: use is_rotating_object
            self.is_rotating_object = False
            self.rotate_axis = None
            print("Finished rotating object")
            event.accept()
        elif event.button() == Qt.LeftButton and self.is_box_selecting:
            self.end_box_selection(event.pos())
            event.accept()
        elif event.button() == Qt.LeftButton and self.current_tool == 'measure':  # UNIVERSAL MEASUREMENT TOOL
            # Directly call measurement tool with screen coordinates
            if self.measurement_tool.handle_release(event.pos().x(), event.pos().y()):
                self.render_window.Render()
            event.accept()
        elif event.button() == Qt.MiddleButton:
            self.is_rotating_view = False
            self.setCursor(Qt.OpenHandCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)
            
    def handle_move_start(self, mouse_pos):
        """Start moving object with gizmo OR free-drag in the view plane."""
        renderer_size = self.renderer.GetSize()
        if renderer_size[0] == 0 or renderer_size[1] == 0:
            return

        # Convert to VTK coordinates (note the flipped Y for VTK)
        x = mouse_pos.x()
        y = renderer_size[1] - mouse_pos.y()

        # 1) Try axis-constrained move first (existing behaviour)
        axis = self.object_manager.move_gizmo.get_axis_at_position(x, y)

        if axis and self.object_manager.selected_actors:
            self.is_moving = True
            self.move_axis = axis
            self.move_start_point = mouse_pos

            # free-move state not used in axis mode
            self.free_move_mouse_world_start = None
            self.free_move_actor_start = None

            print(f"Started moving along {axis.upper()} axis")
            self.render_window.Render()
            return

        # 2) If not on gizmo, try FREE MOVE by dragging the selected actor itself
        picked_actor = self.object_manager.get_object_at_position(x, y)

        if picked_actor is not None and picked_actor in self.object_manager.selected_actors:
            world_pos = self.measurement_tool.get_world_position_from_mouse(x, y)
            if world_pos is None:
                # Fallback to selection if we can't compute world pos
                self.handle_object_selection(mouse_pos, False)
                return

            self.is_moving = True
            self.move_axis = None        # special value: free-move mode
            self.move_start_point = mouse_pos

            self.free_move_mouse_world_start = world_pos
            self.free_move_actor_start = list(picked_actor.GetPosition())

            print("Started FREE move (screen-plane drag)")
            self.render_window.Render()
        else:
            # 3) Otherwise just handle normal selection
            self.handle_object_selection(mouse_pos, False)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.is_scaling and (event.buttons() & Qt.LeftButton):
            self.handle_scale_drag(event.pos())
            event.accept()
        elif self.is_moving and (event.buttons() & Qt.LeftButton):
            self.handle_move_drag(event.pos())
            event.accept()
        elif self.is_rotating_object and (event.buttons() & Qt.LeftButton):  # Fixed: use is_rotating_object
            self.handle_rotate_drag(event.pos())
            event.accept()
        elif self.is_box_selecting and (event.buttons() & Qt.LeftButton):
            self.update_box_selection(event.pos())
            event.accept()
        elif self.current_tool == 'measure' and (event.buttons() & Qt.LeftButton):  # UNIVERSAL MEASUREMENT TOOL
            # Directly call measurement tool with screen coordinates
            if self.measurement_tool.handle_drag(event.pos().x(), event.pos().y()):
                self.render_window.Render()
            event.accept()
        elif self.is_rotating_view and (event.buttons() & Qt.MiddleButton):  # This should work now
            delta = event.pos() - self.last_mouse_pos
            self.last_mouse_pos = event.pos()
            
            # Check if Shift is pressed for panning
            if event.modifiers() & Qt.ShiftModifier:
                # ... existing panning code ...
                pan_sensitivity = 0.005
                
                # Get camera vectors
                camera_pos = self.camera.GetPosition()
                focal_point = self.camera.GetFocalPoint()
                view_up = self.camera.GetViewUp()
                
                # Calculate view direction
                view_dir = [
                    focal_point[0] - camera_pos[0],
                    focal_point[1] - camera_pos[1],
                    focal_point[2] - camera_pos[2]
                ]
                
                # Calculate right vector
                right_vec = [
                    view_up[1] * view_dir[2] - view_up[2] * view_dir[1],
                    view_up[2] * view_dir[0] - view_up[0] * view_dir[2],
                    view_up[0] * view_dir[1] - view_up[1] * view_dir[0]
                ]
                
                # Normalize right vector
                right_length = math.sqrt(right_vec[0]**2 + right_vec[1]**2 + right_vec[2]**2)
                if right_length > 0:
                    right_vec = [right_vec[0]/right_length, right_vec[1]/right_length, right_vec[2]/right_length]
                
                # Calculate pan displacement
                pan_scale = self.camera_radius * 0.1
                pan_x = delta.x() * pan_sensitivity * pan_scale
                pan_y = delta.y() * pan_sensitivity * pan_scale
                
                # Apply pan
                self.rotation_pivot[0] += right_vec[0] * pan_x + view_up[0] * pan_y
                self.rotation_pivot[1] += right_vec[1] * pan_x + view_up[1] * pan_y
                self.rotation_pivot[2] += right_vec[2] * pan_x + view_up[2] * pan_y
                
            else:
                # Regular rotation
                sensitivity = 0.008
                self.camera_phi -= delta.x() * sensitivity
                self.camera_theta -= delta.y() * sensitivity
                self.camera_theta = max(0.01, min(math.pi - 0.01, self.camera_theta))
                self.camera_phi = self.camera_phi % (2 * math.pi)
            
            self.update_camera_position()
            event.accept()
        else:
            super().mouseMoveEvent(event)
            
    def handle_rotate_start(self, mouse_pos):
        """Start rotating object with gizmo"""
        renderer_size = self.renderer.GetSize()
        if renderer_size[0] == 0 or renderer_size[1] == 0:
            return
        
        # Convert to VTK coordinates
        x = mouse_pos.x()
        y = renderer_size[1] - mouse_pos.y()
        
        # Check if clicking on rotate gizmo axis
        axis = self.object_manager.rotate_gizmo.get_axis_at_position(x, y)
        
        if axis and self.object_manager.selected_actors:
            self.is_rotating_object = True  # Fixed: use is_rotating_object
            self.rotate_axis = axis
            self.rotate_start_point = mouse_pos
            print(f"Started rotating around {axis.upper()} axis")
            self.render_window.Render()
        else:
            # If not clicking on gizmo, try to select object
            self.handle_object_selection(mouse_pos, False)

    def handle_rotate_drag(self, mouse_pos):
        """Drag to rotate object with gizmo"""
        if not self.is_rotating_object or not self.object_manager.selected_actors:
            return
        
        selected_actor = self.object_manager.selected_actors[0]
        delta = mouse_pos - self.rotate_start_point
        
        # Convert pixel delta to rotation angle
        sensitivity = 0.5  # degrees per pixel
        rotation_delta = [0, 0, 0]
        
        # Apply rotation based on selected axis
        if self.rotate_axis == 'x':
            rotation_delta[0] = delta.y() * sensitivity
        elif self.rotate_axis == 'y':
            rotation_delta[1] = delta.x() * sensitivity
        elif self.rotate_axis == 'z':
            rotation_delta[2] = delta.x() * sensitivity
        
        # Get current orientation and apply rotation
        current_orientation = selected_actor.GetOrientation()
        new_orientation = [
            current_orientation[0] + rotation_delta[0],
            current_orientation[1] + rotation_delta[1],
            current_orientation[2] + rotation_delta[2]
        ]
        
        # Rotate the object
        selected_actor.SetOrientation(new_orientation)
        
        # Update the outline to match the new orientation in real-time
        if selected_actor in self.object_manager.outline_actors:
            outline_actor = self.object_manager.outline_actors[selected_actor]
            outline_actor.SetOrientation(new_orientation)
            outline_actor.SetPosition(selected_actor.GetPosition())
        
        # Update start point for next rotation
        self.rotate_start_point = mouse_pos
        
        self.render_window.Render()
        
    def handle_scale_start(self, mouse_pos):
        """Start scaling object with gizmo"""
        renderer_size = self.renderer.GetSize()
        if renderer_size[0] == 0 or renderer_size[1] == 0:
            return
        
        # Convert to VTK coordinates
        x = mouse_pos.x()
        y = renderer_size[1] - mouse_pos.y()
        
        # Check if clicking on scale gizmo handle
        axis = self.object_manager.scale_gizmo.get_axis_at_position(x, y)
        
        if axis and self.object_manager.selected_actors:
            self.is_scaling = True
            self.scale_axis = axis
            self.scale_start_point = mouse_pos
            
            # Store original scale
            selected_actor = self.object_manager.selected_actors[0]
            self.original_scale = selected_actor.GetScale()
            
            # PAUSE property panel updates during scaling
            if hasattr(self, 'main_window') and self.main_window:
                self.main_window.right_panel.transform_widget.pause_updates()
            
            print(f"Started scaling on {axis.upper()} axis")
            self.render_window.Render()
        else:
            # If not clicking on gizmo, try to select object
            self.handle_object_selection(mouse_pos, False)
            
    def handle_measure_start(self, mouse_pos):
        """Start measurement tool interaction"""
        world_pos = self.get_world_position_from_mouse(mouse_pos)
        if world_pos:
            if self.measurement_tool.handle_click(world_pos):
                self.render_window.Render()
            else:
                # If not handling measurement click, fall back to selection
                self.handle_object_selection(mouse_pos, False)
    
    def handle_scale_drag(self, mouse_pos):
        """Drag to scale object with gizmo"""
        if not self.is_scaling or not self.object_manager.selected_actors:
            return
        
        selected_actor = self.object_manager.selected_actors[0]
        delta = mouse_pos - self.scale_start_point
        
        sensitivity = 0.01
        scale_factor = 1.0 + delta.x() * sensitivity  # Use horizontal movement for scaling
        
        new_scale = list(self.original_scale)  # Start with original scale
        
        if self.scale_axis == 'x':
            new_scale[0] = self.original_scale[0] * scale_factor
        elif self.scale_axis == 'y':
            new_scale[1] = self.original_scale[1] * scale_factor
        elif self.scale_axis == 'z':
            new_scale[2] = self.original_scale[2] * scale_factor
        elif self.scale_axis == 'uniform':
            # Uniform scaling on all axes
            new_scale[0] = self.original_scale[0] * scale_factor
            new_scale[1] = self.original_scale[1] * scale_factor
            new_scale[2] = self.original_scale[2] * scale_factor
        
        # Apply scaling (minimum scale of 0.1 to prevent inversion)
        new_scale = [max(0.1, s) for s in new_scale]
        selected_actor.SetScale(new_scale)
        
        # Update outline to match the new scale
        if selected_actor in self.object_manager.outline_actors:
            outline_actor = self.object_manager.outline_actors[selected_actor]
            outline_actor.SetScale(new_scale)
        
        self.render_window.Render()
            
    def wheelEvent(self, event):
        # Ensure the VTK widget doesn't steal focus
        self.vtk_widget.clearFocus()
        self.setFocus()
        
        # Blender-like zoom (towards/away from focal point)
        zoom_factor = 1.1
        
        if event.angleDelta().y() > 0:
            # Zoom in
            self.camera_radius /= zoom_factor
        else:
            # Zoom out
            self.camera_radius *= zoom_factor
        
        # Set reasonable bounds (like Blender's clip range)
        self.camera_radius = max(0.1, min(1000.0, self.camera_radius))
        
        # Update immediately
        self.update_camera_position()
        event.accept()
        
    def handle_move_drag(self, mouse_pos):
        """Drag object with gizmo or free-drag in view plane."""
        if not self.is_moving or not self.object_manager.selected_actors:
            return

        selected_actor = self.object_manager.selected_actors[0]

        # -------- FREE-MOVE MODE (move_axis is None) --------
        if self.move_axis is None:
            renderer_size = self.renderer.GetSize()
            if renderer_size[0] == 0 or renderer_size[1] == 0:
                return

            if (
                self.free_move_mouse_world_start is None or
                self.free_move_actor_start is None
            ):
                return

            # Current mouse -> world position (using the measurement tool)
            x = mouse_pos.x()
            y = renderer_size[1] - mouse_pos.y()
            world_now = self.measurement_tool.get_world_position_from_mouse(x, y)
            if world_now is None:
                return

            # Delta in world space (view plane)
            world_delta = [
                world_now[0] - self.free_move_mouse_world_start[0],
                world_now[1] - self.free_move_mouse_world_start[1],
                world_now[2] - self.free_move_mouse_world_start[2],
            ]

            new_pos = [
                self.free_move_actor_start[0] + world_delta[0],
                self.free_move_actor_start[1] + world_delta[1],
                self.free_move_actor_start[2] + world_delta[2],
            ]

            # Move actor directly to new position
            selected_actor.SetPosition(new_pos)

            # Keep outline in sync if present
            if selected_actor in self.object_manager.outline_actors:
                outline_actor = self.object_manager.outline_actors[selected_actor]
                outline_actor.SetPosition(new_pos)

            # Update gizmo position to follow
            self.object_manager.move_gizmo.update_position()

            self.render_window.Render()
            return

        # -------- ORIGINAL AXIS-CONSTRAINED MODE --------
        delta = mouse_pos - self.move_start_point

        sensitivity = 0.01
        world_delta = [0, 0, 0]

        # MIXED with inverted directions for red and blue (your current behaviour)
        if self.move_axis == 'x':
            # Red arrow (horizontal): Use horizontal mouse movement INVERTED
            world_delta[0] = -delta.x() * sensitivity  # INVERTED
        elif self.move_axis == 'y':
            # Green arrow (into screen): Use horizontal mouse movement
            world_delta[1] = delta.x() * sensitivity
        elif self.move_axis == 'z':
            # Blue arrow (vertical): Use vertical mouse movement INVERTED
            world_delta[2] = -delta.y() * sensitivity  # INVERTED

        print(
            f"Moving {self.move_axis} axis: "
            f"delta=({delta.x()}, {delta.y()}), "
            f"world_delta=({world_delta[0]:.3f}, {world_delta[1]:.3f}, {world_delta[2]:.3f})"
        )

        # Move the object using ObjectManager
        self.object_manager.move_object(
            selected_actor, world_delta[0], world_delta[1], world_delta[2]
        )

        # Update gizmo position
        self.object_manager.move_gizmo.update_position()

        # Update start point for next movement (axis mode)
        self.move_start_point = mouse_pos

        self.render_window.Render()
    
    def set_rotation_pivot(self, x, y, z):
        """Set the rotation pivot point (like Blender's 3D cursor)"""
        self.rotation_pivot = [x, y, z]
        self.update_camera_position()
    
    def reset_view(self):
        """Reset to default view (like Blender's Home key)"""
        self.camera_radius = 15.0
        self.camera_theta = math.radians(45.0)
        self.camera_phi = math.radians(45.0)
        self.rotation_pivot = [0, 0, 0]
        self.update_camera_position()
        
    def handle_object_selection(self, mouse_pos, multi_select=False):
        """Handle left-click object selection with multi-select support"""
        if not self.object_manager.actors:
            return
        
        renderer_size = self.renderer.GetSize()
        if renderer_size[0] == 0 or renderer_size[1] == 0:
            return
        
        # Convert to VTK coordinates
        x = mouse_pos.x()
        y = renderer_size[1] - mouse_pos.y()
        
        # Pick object at mouse position
        picked_actor = self.object_manager.get_object_at_position(x, y)
        
        if picked_actor:
            # Select the clicked object
            self.object_manager.select_object(picked_actor, multi_select)
        else:
            # Clicked on empty space - deselect all
            if not multi_select:  # Only deselect if not multi-selecting
                self.object_manager.deselect_all()
        
        self.update_camera_position()
    
    def create_object(self, object_type):
        """Create a 3D object"""
        self.object_manager.create_object(object_type)
        self.update_camera_position()
        
    def create_light(self, light_type: str):
        """Create a light via LightManager."""
        if not hasattr(self, "light_manager"):
            return
        actor = self.light_manager.create_light(light_type)
        # Center camera if you want:
        self.update_camera_position()
        self.render_window.Render()
    
    def change_color(self, color):
        """Change object color"""
        self.object_manager.change_color(color)
        self.update_camera_position()
        
    def start_box_selection(self, pos):
        """Start box selection with debug info"""
        print(f"=== Starting Box Selection ===")
        print(f"Mouse position: {pos.x()}, {pos.y()}")
        
        self.is_box_selecting = True
        self.box_select_start = pos
        
        # Convert to VTK widget coordinates
        vtk_pos = self.vtk_widget.mapFrom(self, pos)
        print(f"Mapped to VTK widget: {vtk_pos.x()}, {vtk_pos.y()}")
        
        self.box_select_rubberband.start_selection(vtk_pos)
        print("Rubber band started")
    
    def update_box_selection(self, pos):
        """Update box selection"""
        # Convert to VTK widget coordinates
        vtk_pos = self.vtk_widget.mapFrom(self, pos)
        self.box_select_rubberband.update_selection(vtk_pos)
        
    def end_box_selection(self, pos):
        """End box selection with debug info"""
        print(f"=== Ending Box Selection ===")
        print(f"End position: {pos.x()}, {pos.y()}")
        
        self.is_box_selecting = False
        self.box_select_rubberband.end_selection()
        
        # Calculate selection box
        x1 = min(self.box_select_start.x(), pos.x())
        y1 = min(self.box_select_start.y(), pos.y())
        x2 = max(self.box_select_start.x(), pos.x())
        y2 = max(self.box_select_start.y(), pos.y())
        
        print(f"Selection box: ({x1}, {y1}) to ({x2}, {y2})")
        print(f"Selection size: {x2-x1} x {y2-y1}")
        
        # Select objects in the box
        self.select_objects_in_box(x1, y1, x2, y2)
    
    def select_objects_in_box(self, x1, y1, x2, y2):
        """Select all objects that fall within the screen rectangle using proper picking"""
        renderer_size = self.renderer.GetSize()
        if renderer_size[0] == 0 or renderer_size[1] == 0:
            print("Renderer has zero size")
            return
        
        print(f"Selection box: ({x1}, {y1}) to ({x2}, {y2})")
        print(f"Renderer size: {renderer_size}")
        
        # Convert to VTK display coordinates (origin at bottom-left)
        display_x1 = x1
        display_y1 = renderer_size[1] - y2  # Flip Y
        display_x2 = x2  
        display_y2 = renderer_size[1] - y1  # Flip Y
        
        print(f"VTK display: ({display_x1}, {display_y1}) to ({display_x2}, {display_y2})")
        
        selected_actors = []
        
        # METHOD 1: Use vtkHardwareSelector for reliable picking
        try:
            selector = vtk.vtkHardwareSelector()
            selector.SetRenderer(self.renderer)
            selector.SetArea(display_x1, display_y1, display_x2, display_y2)
            selector.SetFieldAssociation(vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS)
            
            selection = selector.Select()
            
            if selection:
                node_count = selection.GetNumberOfNodes()
                print(f"Hardware selector found {node_count} nodes")
                
                for i in range(node_count):
                    node = selection.GetNode(i)
                    if node:
                        prop = node.GetProp()
                        if isinstance(prop, vtk.vtkActor) and prop in self.object_manager.actors:
                            if prop not in selected_actors:
                                selected_actors.append(prop)
                                print(f"Selected actor via hardware picker")
            
            selection.UnRegister(selector)
            
        except Exception as e:
            print(f"Hardware selector failed: {e}")
        
        # METHOD 2: Fallback to simple bounds checking if hardware picking fails
        if not selected_actors:
            print("Trying bounds-based selection...")
            for i, actor in enumerate(self.object_manager.actors):
                # Get actor bounds in world coordinates
                bounds = actor.GetBounds()
                if not bounds:
                    continue
                    
                # Calculate center of the actor
                center = [
                    (bounds[0] + bounds[1]) / 2,
                    (bounds[2] + bounds[3]) / 2, 
                    (bounds[4] + bounds[5]) / 2
                ]
                
                # Convert world center to display coordinates
                display_coords = [0, 0, 0]
                self.renderer.SetWorldPoint(center[0], center[1], center[2], 1.0)
                self.renderer.WorldToDisplay()
                self.renderer.GetDisplayPoint(display_coords)
                
                screen_x = display_coords[0]
                screen_y = renderer_size[1] - display_coords[1]  # Flip back to Qt coords
                
                print(f"Actor {i} screen position: ({screen_x}, {screen_y})")
                
                # Check if center is within selection box with some tolerance
                tolerance = 10  # pixels
                if (x1 - tolerance <= screen_x <= x2 + tolerance and 
                    y1 - tolerance <= screen_y <= y2 + tolerance):
                    selected_actors.append(actor)
                    print(f"Actor {i} selected via bounds check")
        
        # METHOD 3: Last resort - select all objects if large area selected
        if not selected_actors and (x2 - x1 > 100 and y2 - y1 > 100):
            print("Large selection area - selecting all objects")
            selected_actors = self.object_manager.actors.copy()
        
        # Apply the selection
        if selected_actors:
            self.object_manager.select_objects_in_area(selected_actors)
            status_msg = f"Selected {len(selected_actors)} objects"
            print(status_msg)
        else:
            self.object_manager.deselect_all()
            status_msg = "No objects in selection area"
            print(status_msg)
        
        if self.main_window:
            self.main_window.statusBar().showMessage(status_msg)
        
        self.update_camera_position()
        
    def debug_gizmo_status(self):
        """Debug method to check gizmo status"""
        print("=== GIZMO DEBUG INFO ===")
        print(f"Current tool: {self.current_tool}")
        print(f"Selected actors: {len(self.object_manager.selected_actors)}")
        print(f"Active gizmo: {self.object_manager.active_gizmo}")
        print(f"Move gizmo visible: {self.object_manager.move_gizmo.is_visible}")
        print(f"Rotate gizmo visible: {self.object_manager.rotate_gizmo.is_visible}")
        
        # Check if gizmo actors are in renderer
        move_actors_in_renderer = 0
        rotate_actors_in_renderer = 0
        
        actors = self.renderer.GetActors()
        actors.InitTraversal()
        for i in range(actors.GetNumberOfItems()):
            actor = actors.GetNextActor()
            if actor in self.object_manager.move_gizmo.actors.values():
                move_actors_in_renderer += 1
            if actor in self.object_manager.rotate_gizmo.actors.values():
                rotate_actors_in_renderer += 1
        
        print(f"Move gizmo actors in renderer: {move_actors_in_renderer}")
        print(f"Rotate gizmo actors in renderer: {rotate_actors_in_renderer}")
        print("=== END DEBUG INFO ===")
        
    def set_main_window(self, main_window):
        """Set reference to main window for status updates"""
        self.main_window = main_window
    
    def on_tool_changed(self, tool_name):
        """Handle tool changes from the toolbar"""
        self.current_tool = tool_name
        print(f"Active tool: {tool_name}")
        
        if tool_name != 'measure':
            self.measurement_tool.deactivate()
        
        # Update cursor and gizmo visibility
        if tool_name == 'select':
            self.setCursor(Qt.ArrowCursor)
            # Hide all gizmos when switching to select tool
            self.object_manager.move_gizmo.hide()
            self.object_manager.rotate_gizmo.hide()
            # self.measurement_tool.deactivate()  # DEACTIVATE MEASUREMENT
        elif tool_name == 'move':
            self.setCursor(Qt.SizeAllCursor)
            # Show move gizmo and hide rotate gizmo
            self.object_manager.rotate_gizmo.hide()
            # self.measurement_tool.deactivate()  # DEACTIVATE MEASUREMENT
            if self.object_manager.selected_actors:
                print("Calling move gizmo show from on_tool_changed")
                self.object_manager.move_gizmo.show(self.object_manager.selected_actors[0])
            else:
                print("No selected object for move gizmo")
        elif tool_name == 'rotate':
            self.setCursor(Qt.CrossCursor)
            # Show rotate gizmo and hide move gizmo
            self.object_manager.move_gizmo.hide()
            # self.measurement_tool.deactivate()  # DEACTIVATE MEASUREMENT
            if self.object_manager.selected_actors:
                print("Calling rotate gizmo show from on_tool_changed")
                self.object_manager.rotate_gizmo.show(self.object_manager.selected_actors[0])
            else:
                print("No selected object for rotate gizmo")
        elif tool_name == 'box_select':
            self.setCursor(Qt.CrossCursor)
            # Hide all gizmos
            self.object_manager.move_gizmo.hide()
            self.object_manager.rotate_gizmo.hide()
            # self.measurement_tool.deactivate()  # DEACTIVATE MEASUREMENT
        elif tool_name == 'measure':  # UNIVERSAL MEASUREMENT TOOL
            self.setCursor(Qt.CrossCursor)
            self.object_manager.move_gizmo.hide()
            self.object_manager.rotate_gizmo.hide()
            self.object_manager.scale_gizmo.hide()
            self.measurement_tool.activate()
            if self.main_window:
                self.main_window.statusBar().showMessage("Measurement Tool: Click and drag ANYWHERE in 3D space to measure distance")
        
        # CRITICAL: Update the object manager's active tool
        self.object_manager.set_active_tool(tool_name)
        
        # Debug info
        self.debug_gizmo_status()
        
        # Force a render to update the display
        self.render_window.Render()
    
    def end_box_selection(self, pos):
        """End box selection and select objects in the box"""
        self.is_box_selecting = False
        self.box_select_rubberband.end_selection()
        
        # Calculate selection box
        x1 = min(self.box_select_start.x(), pos.x())
        y1 = min(self.box_select_start.y(), pos.y())
        x2 = max(self.box_select_start.x(), pos.x())
        y2 = max(self.box_select_start.y(), pos.y())
        
        print(f"Box selection: ({x1}, {y1}) to ({x2}, {y2})")
        
        # Update status bar through main window
        if self.main_window:
            self.main_window.statusBar().showMessage(f"Box selection: {x2-x1}x{y2-y1} pixels")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Blender-like 3D Viewer with Object Controls")
        self.setGeometry(100, 100, 1400, 900)
        
        # Apply dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QStatusBar {
                background-color: #3c3c3c;
                color: #ffffff;
            }
            QMenuBar {
                background-color: #3c3c3c;
                color: #ffffff;
            }
            QMenuBar::item:selected {
                background-color: #505050;
            }
            QMenu {
                background-color: #3c3c3c;
                color: #ffffff;
            }
            QMenu::item:selected {
                background-color: #505050;
            }
            QLineEdit {
                background-color: #2b2b2b;
                color: white;
                border: 1px solid #404040;
                border-radius: 3px;
                padding: 3px;
            }
            QLineEdit:focus {
                border: 1px solid #ff8000;
            }
            QLabel {
                color: #cccccc;
}
        """)
        
        # Create left toolbar
        self.left_toolbar = LeftToolbar(self)
        self.addToolBar(Qt.LeftToolBarArea, self.left_toolbar)
        
        # Create right panel
        self.right_panel = RightPanel(self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.right_panel)
        
        # NEW: Light panel (right side, scrollable if needed)
        self.light_panel = LightPanel(self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.light_panel)
        
        # NEW: Create object creation panel
        self.object_creation_panel = ObjectCreationPanel(self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.object_creation_panel)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create layout
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create VTK widget
        self.vtk_widget = VTKWidget()
        self.vtk_widget.set_main_window(self)
        layout.addWidget(self.vtk_widget)
        
        self.right_panel.set_vtk_widget(self.vtk_widget)
        
        # Connect panels to VTK widget
        # self.properties_panel.set_vtk_widget(self.vtk_widget)
        self.object_creation_panel.set_vtk_widget(self.vtk_widget)
        
        # NEW: connect lights panel
        self.light_panel.set_vtk_widget(self.vtk_widget)
        
        # Connect toolbar to VTK widget
        self.left_toolbar.parent_widget = self.vtk_widget
        
        # Create 2D viewport gizmo
        self.viewport_gizmo = ViewportGizmo2D()
        self.viewport_gizmo.set_main_window(self)
        self.viewport_gizmo.show()
            
        # Timer to update gizmo orientation
        self.gizmo_timer = QTimer()
        self.gizmo_timer.timeout.connect(self.update_gizmo)
        self.gizmo_timer.start(50)
        
        # Create menu bar
        self.create_menu()
        
        # Create status bar with Blender-like instructions
        self.statusBar().showMessage("MMB: Orbit | Shift+MMB: Pan | Wheel: Zoom | Click Gizmo: Reset View")
    
    def create_menu(self):
        """Create the interactive menu bar"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('File')
        
        # Load Model submenu
        load_model_menu = QMenu('Load Model', self)
        
        # Add file format options
        obj_action = QAction('OBJ File', self)
        obj_action.setStatusTip('Load OBJ 3D model file')
        obj_action.triggered.connect(lambda: self.load_model('obj'))
        
        ply_action = QAction('PLY File', self)
        ply_action.setStatusTip('Load PLY 3D model file')
        ply_action.triggered.connect(lambda: self.load_model('ply'))
        
        stl_action = QAction('STL File', self)
        stl_action.setStatusTip('Load STL 3D model file')
        stl_action.triggered.connect(lambda: self.load_model('stl'))
        
        three_ds_action = QAction('3DS File', self)
        three_ds_action.setStatusTip('Load 3DS 3D model file')
        three_ds_action.triggered.connect(lambda: self.load_model('3ds'))
        
        # Add actions to load model submenu
        load_model_menu.addAction(obj_action)
        load_model_menu.addAction(ply_action)
        load_model_menu.addAction(stl_action)
        load_model_menu.addAction(three_ds_action)
        
        # Add load model submenu to file menu
        file_menu.addMenu(load_model_menu)
        
        # Separator
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Object menu - for changing 3D objects
        object_menu = menubar.addMenu('Object')
        
        sphere_action = QAction('Display Sphere', self)
        sphere_action.setStatusTip('Display a sphere')
        sphere_action.triggered.connect(lambda: self.vtk_widget.create_object('sphere'))

        cube_action = QAction('Display Cube', self)
        cube_action.setStatusTip('Display a cube')
        cube_action.triggered.connect(lambda: self.vtk_widget.create_object('cube'))

        object_menu.addAction(sphere_action)
        object_menu.addAction(cube_action)

        # Color menu - for changing object colors
        color_menu = menubar.addMenu('Color')
        
        white_action = QAction('White', self)
        white_action.setStatusTip('Set color to white')
        white_action.triggered.connect(lambda: self.vtk_widget.change_color((1.0, 1.0, 1.0)))

        red_action = QAction('Red', self)
        red_action.setStatusTip('Set color to red')
        red_action.triggered.connect(lambda: self.vtk_widget.change_color((1.0, 0.0, 0.0)))

        green_action = QAction('Green', self)
        green_action.setStatusTip('Set color to green')
        green_action.triggered.connect(lambda: self.vtk_widget.change_color((0.0, 1.0, 0.0)))

        blue_action = QAction('Blue', self)
        blue_action.setStatusTip('Set color to blue')
        blue_action.triggered.connect(lambda: self.vtk_widget.change_color((0.0, 0.0, 1.0)))

        color_menu.addAction(white_action)
        color_menu.addAction(red_action)
        color_menu.addAction(green_action)
        color_menu.addAction(blue_action)
        
                # NEW: Save View submenu
        save_view_menu = QMenu('Save View', self)
        
        save_current_view_action = QAction('Save Current View as Image', self)
        save_current_view_action.setStatusTip('Save the current view as an image file')
        save_current_view_action.triggered.connect(self.save_current_view)
        
        save_camera_view_action = QAction('Save Camera View as Image', self)
        save_camera_view_action.setStatusTip('Save a selected camera view as an image file')
        save_camera_view_action.triggered.connect(self.save_camera_view_from_menu)
        
        save_view_menu.addAction(save_current_view_action)
        save_view_menu.addAction(save_camera_view_action)
        
        file_menu.addMenu(load_model_menu)
        file_menu.addMenu(save_view_menu)  # NEW: Add save view menu
        file_menu.addSeparator()
        
    def load_model(self, file_type):
        """Load a 3D model file"""
        # File type filters
        file_filters = {
            'obj': 'OBJ Files (*.obj)',
            'ply': 'PLY Files (*.ply)',
            'stl': 'STL Files (*.stl)',
            '3ds': '3DS Files (*.3ds)'
        }
        
        filter_str = file_filters.get(file_type, 'All Files (*.*)')
        
        # Open file dialog
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f'Load {file_type.upper()} Model',
            '',
            f'{filter_str};;All Files (*.*)'
        )
        
        if file_path:
            print(f"Loading {file_type} file: {file_path}")
            success = self.vtk_widget.load_external_model(file_path, file_type)
            
            if success:
                self.statusBar().showMessage(f"Successfully loaded: {file_path}")
            else:
                self.statusBar().showMessage(f"Failed to load: {file_path}")
                QMessageBox.warning(self, "Load Error", f"Could not load {file_type.upper()} file: {file_path}")
    
    def update_gizmo(self):
        """Update both gizmo orientation and position"""
        # Update orientation
        self.viewport_gizmo.update_orientation(
            self.vtk_widget.camera_phi,
            self.vtk_widget.camera_theta
        )
        # Update position (in case window was resized or panel moved)
        self.viewport_gizmo.update_position()
    
    def resizeEvent(self, event):
        """Update gizmo position when window is resized"""
        super().resizeEvent(event)
        if hasattr(self, 'viewport_gizmo'):
            self.viewport_gizmo.update_position()
            
    def moveEvent(self, event):
        """Update gizmo position when window is moved"""
        super().moveEvent(event)
        if hasattr(self, 'viewport_gizmo'):
            self.viewport_gizmo.update_position()
            
    def on_tool_changed(self, tool_name):
        """Forward tool changes to VTK widget"""
        self.vtk_widget.on_tool_changed(tool_name)
        
    def save_current_view(self):
        """Save the current view as an image"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Current View as Image",
            f"view_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            "PNG Images (*.png);;JPEG Images (*.jpg *.jpeg);;All Files (*)"
        )
        
        if file_path and hasattr(self, 'vtk_widget'):
            success = self.vtk_widget.save_view_image(file_path)
            if success:
                QMessageBox.information(self, "Success", f"Current view saved as:\n{file_path}")
            else:
                QMessageBox.warning(self, "Error", "Failed to save view image.")

    def save_camera_view_from_menu(self):
        """Save camera view from menu (triggers the right panel button)"""
        if hasattr(self, 'right_panel'):
            self.right_panel.save_camera_view()

def main():
    # Create application
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Create and show main window
    window = MainWindow()
    window.showMaximized()  # Start maximized
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()