# ------------------------------------------------------------------------
#    COLMAP code: https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
# ------------------------------------------------------------------------

import os
import collections
import numpy as np
import struct
import argparse

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras


def read_cameras_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8 * num_params,
                                     format_char_sequence="d" * num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras


def read_images_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images


def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24 * num_points2D,
                                       format_char_sequence="ddq" * num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    points3D = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                image_ids = np.array(tuple(map(int, elems[8::2])))
                point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                points3D[point3D_id] = Point3D(id=point3D_id, xyz=xyz, rgb=rgb,
                                               error=error, image_ids=image_ids,
                                               point2D_idxs=point2D_idxs)
    return points3D


def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id, xyz=xyz, rgb=rgb,
                error=error, image_ids=image_ids,
                point2D_idxs=point2D_idxs)
    return points3D


def detect_model_format(path, ext):
    if os.path.isfile(os.path.join(path, "cameras" + ext)) and \
            os.path.isfile(os.path.join(path, "images" + ext)) and \
            os.path.isfile(os.path.join(path, "points3D" + ext)):
        print("Detected model format: '" + ext + "'")
        return True

    return False


def read_model(path, ext=""):
    # try to detect the extension automatically
    if ext == "":
        if detect_model_format(path, ".bin"):
            ext = ".bin"
        elif detect_model_format(path, ".txt"):
            ext = ".txt"
        else:
            print("Provide model format: '.bin' or '.txt'")
            return

    if ext == ".txt":
        cameras = read_cameras_text(os.path.join(path, "cameras" + ext))
        images = read_images_text(os.path.join(path, "images" + ext))
        points3D = read_points3D_text(os.path.join(path, "points3D") + ext)
    else:
        cameras = read_cameras_binary(os.path.join(path, "cameras" + ext))
        images = read_images_binary(os.path.join(path, "images" + ext))
        points3D = read_points3D_binary(os.path.join(path, "points3D") + ext)
    return cameras, images, points3D


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


# ------------------------------------------------------------------------
#    AddOn code:
#    useful tutorial: https://blender.stackexchange.com/questions/57306/how-to-create-a-custom-ui
# ------------------------------------------------------------------------

# bl_info
bl_info = {
    "name": "BlenderNeuralangelo",
    "version": (1, 0),
    "blender": (3, 3, 1),
    "location": "PROPERTIES",
    "warning": "",  # used for warning icon and text in addons panel
    "support": "COMMUNITY",
    "category": "Render"
}

import bpy
from bpy.props import (StringProperty,
                       BoolProperty,
                       IntProperty,
                       FloatProperty,
                       FloatVectorProperty,
                       EnumProperty,
                       PointerProperty,
                       )
from bpy.types import (Panel,
                       Menu,
                       Operator,
                       PropertyGroup,
                       )

# global variable for easier access
colmap_data = {}


# ------------------------------------------------------------------------
#    Utility scripts
# ------------------------------------------------------------------------

def display_pointcloud(points3D):
    '''
    load and display point cloud
    borrowed from https://github.com/TombstoneTumbleweedArt/import-ply-as-verts
    '''

    xyzs = np.stack([point.xyz for point in points3D.values()])
    rgbs = np.stack([point.rgb for point in points3D.values()])  # / 255.0

    # Copy the positions
    ply_name = 'point cloud'
    mesh = bpy.data.meshes.new(name=ply_name)
    mesh.vertices.add(xyzs.shape[0])
    mesh.vertices.foreach_set("co", [a for v in xyzs for a in v])

    # Create our new object here
    for ob in bpy.context.selected_objects:
        ob.select_set(False)
    obj = bpy.data.objects.new(ply_name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    mesh.update()
    mesh.validate()


def generate_cropping_planes():
    points3D = colmap_data['points3D']
    xyzs = np.stack([point.xyz for point in points3D.values()])

    x_min = min(xyzs[:, 0])
    x_max = max(xyzs[:, 0])
    y_min = min(xyzs[:, 1])
    y_max = max(xyzs[:, 1])
    z_min = min(xyzs[:, 2])
    z_max = max(xyzs[:, 2])

    verts = [[x_max, y_max, z_min],
             [x_max, y_min, z_min],
             [x_min, y_min, z_min],
             [x_min, y_max, z_min],
             [x_max, y_max, z_max],
             [x_max, y_min, z_max],
             [x_min, y_min, z_max],
             [x_min, y_max, z_max]]

    faces = [[0, 1, 5, 4],
             [3, 2, 6, 7],
             [0, 3, 7, 4],
             [1, 2, 6, 5],
             [0, 1, 2, 3],
             [4, 5, 6, 7]]

    msh = bpy.data.meshes.new("cropping plane")
    msh.from_pydata(verts, [], faces)
    obj = bpy.data.objects.new("cropping plane", msh)
    bpy.context.scene.collection.objects.link(obj)

    return


def update_cropping_plane(scene, depsgraph):
    point_cloud = bpy.data.objects['point cloud'].data
    cloud_verts = [v for v in point_cloud.vertices]

    x_min = min(v.co.x for v in cloud_verts)
    x_max = max(v.co.x for v in cloud_verts)
    y_min = min(v.co.y for v in cloud_verts)
    y_max = max(v.co.y for v in cloud_verts)
    z_min = min(v.co.z for v in cloud_verts)
    z_max = max(v.co.z for v in cloud_verts)

    slider = bpy.context.scene.my_tool.my_slider
    crop_plane = bpy.data.objects['cropping plane']

    x_min_change = slider[0]
    x_max_change = slider[1]
    y_min_change = slider[2]
    y_max_change = slider[3]
    z_min_change = slider[4]
    z_max_change = slider[5]

    crop_plane.data.vertices[0].co.x = x_max + x_max_change
    crop_plane.data.vertices[0].co.y = y_max + y_max_change
    crop_plane.data.vertices[0].co.z = z_min - z_min_change

    crop_plane.data.vertices[1].co.x = x_max + x_max_change
    crop_plane.data.vertices[1].co.y = y_min - y_min_change
    crop_plane.data.vertices[1].co.z = z_min - z_min_change

    crop_plane.data.vertices[2].co.x = x_min - x_min_change
    crop_plane.data.vertices[2].co.y = y_min - y_min_change
    crop_plane.data.vertices[2].co.z = z_min - z_min_change

    crop_plane.data.vertices[3].co.x = x_min - x_min_change
    crop_plane.data.vertices[3].co.y = y_max + y_max_change
    crop_plane.data.vertices[3].co.z = z_min - z_min_change

    crop_plane.data.vertices[4].co.x = x_max + x_max_change
    crop_plane.data.vertices[4].co.y = y_max + y_max_change
    crop_plane.data.vertices[4].co.z = z_max + z_max_change

    crop_plane.data.vertices[5].co.x = x_max + x_max_change
    crop_plane.data.vertices[5].co.y = y_min - y_min_change
    crop_plane.data.vertices[5].co.z = z_max + z_max_change

    crop_plane.data.vertices[6].co.x = x_min - x_min_change
    crop_plane.data.vertices[6].co.y = y_min - y_min_change
    crop_plane.data.vertices[6].co.z = z_max + z_max_change

    crop_plane.data.vertices[7].co.x = x_min - x_min_change
    crop_plane.data.vertices[7].co.y = y_max + y_max_change
    crop_plane.data.vertices[7].co.z = z_max + z_max_change


# ------------------------------------------------------------------------
#    Scene Properties
# ------------------------------------------------------------------------

class MyProperties(PropertyGroup):
    '''
    slider bar, path, and everything else ....
    '''
    colmap_path: StringProperty(
        name="Directory",
        description="Choose a directory:",
        default="",
        maxlen=1024,
        subtype='DIR_PATH'
    )
    my_slider: FloatVectorProperty(
        name="Plane offset",
        subtype='TRANSLATION',
        description="X_min, X_max ,Y_min ,Y_max ,Z_min ,Z_max",
        size=6,
        min=-50,
        max=50,
        default=(0, 0, 0, 0, 0, 0),
    )


# ------------------------------------------------------------------------
#    Operators, i.e, buttons + callback
# ------------------------------------------------------------------------

class OT_LoadCOLMAP(Operator):
    bl_label = "Load COLMAP Data"
    bl_idname = "my.load_colmap"

    def execute(self, context):
        scene = context.scene
        mytool = scene.my_tool

        print("loading data")
        cameras, images, points3D = read_model(bpy.path.abspath(mytool.colmap_path + 'sparse/'), ext='.bin')
        display_pointcloud(points3D)

        print("store colmap data")
        global colmap_data
        colmap_data['cameras'] = cameras
        colmap_data['images'] = images
        colmap_data['points3D'] = points3D

        print("TODO: set cropping planes location")
        generate_cropping_planes()

        print("TODO: set camera intrinsics")

        print("TODO: set camera poses")

        print("TODO: load images")

        return {'FINISHED'}


class OT_Debug(Operator):
    '''
    for easier debugging experience
    '''

    bl_label = "Debug"
    bl_idname = "my.debug"

    def execute(self, context):
        scene = context.scene
        mytool = scene.my_tool

        print(len(colmap_data['cameras'].keys()))
        print(len(colmap_data['images'].keys()))
        print(len(colmap_data['points3D'].keys()))

        return {'FINISHED'}


class Crop(Operator):
    '''
    crop points outside the bounding box

    note: if you want to see the result of point cropping, please follow the steps:

            1.click "crop points" button
            2.enter the edit mode
            3.hide the cropping plane

    '''

    bl_label = "Crop Pointcloud"
    bl_idname = "my.crop"

    def execute(self, context):

        plane_verts = bpy.data.objects['cropping plane'].data.vertices
        cloud_verts = bpy.data.objects['point cloud'].data.vertices

        x_min = min(v.co.x for v in plane_verts)
        x_max = max(v.co.x for v in plane_verts)
        y_min = min(v.co.y for v in plane_verts)
        y_max = max(v.co.y for v in plane_verts)
        z_min = min(v.co.z for v in plane_verts)
        z_max = max(v.co.z for v in plane_verts)

        num = 0
        for v in cloud_verts:
            v.hide = False
            if x_min <= v.co.x <= x_max and y_min <= v.co.y <= y_max and z_min <= v.co.z <= z_max:
                num += 1
            else:
                v.hide = True

        print("crop finished; left vertices number :", num)
        return {'FINISHED'}


class BoundSphere(Operator):
    '''
    crop points outside the bounding box
    '''

    bl_label = "Create Bounding Sphere"
    bl_idname = "my.add_bound_sphere"

    def execute(self, context):

        for obj in bpy.context.scene.objects:
            if obj.name == 'Sphere':
                mesh_obj = bpy.data.objects.get("Sphere")
                bpy.data.meshes.remove(mesh_obj.data, do_unlink=True)

        cloud_verts = bpy.data.objects['point cloud'].data.vertices

        unhide_vert = []
        for v in cloud_verts:
            if v.hide == False:
                unhide_vert.append(v)
        print(len(unhide_vert))

        x_min = min(v.co.x for v in unhide_vert)
        x_max = max(v.co.x for v in unhide_vert)
        y_min = min(v.co.y for v in unhide_vert)
        y_max = max(v.co.y for v in unhide_vert)
        z_min = min(v.co.z for v in unhide_vert)
        z_max = max(v.co.z for v in unhide_vert)

        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        center_z = (z_min + z_max) / 2

        Radius = max(np.sqrt((v.co.x - center_x) ** 2 + (v.co.y - center_y) ** 2 + (v.co.z - center_z) ** 2) for v in
                     unhide_vert)

        bpy.ops.mesh.primitive_uv_sphere_add(radius=Radius, location=(center_x, center_y, center_z))

        print("create bounding sphere finished")
        return {'FINISHED'}


class HideShowBox(Operator):
    bl_label = "Hide/Show Bounding Box"
    bl_idname = "my.hide_show_box"

    def execute(self, context):
        status = bpy.context.scene.objects['cropping plane'].hide_get()
        bpy.context.scene.objects['cropping plane'].hide_set(not status)
        return {'FINISHED'}


# ------------------------------------------------------------------------
#    Panel
# ------------------------------------------------------------------------

class NeuralangeloCustomPanel:
    bl_category = "Neuralangelo"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

class MainPanel(NeuralangeloCustomPanel, bpy.types.Panel):
    bl_idname = "panel_main"
    bl_label = "Neuralangelo Addon"
    
    def draw(self, context):
        layout = self.layout
        layout.label("BlenderNeuralangelo")
    
class LoadingPanel(NeuralangeloCustomPanel, bpy.types.Panel):
    bl_parent_id  = "panel_main"
    bl_label = "Load Data"

    def draw(self, context):
        scene = context.scene
        layout = self.layout
        mytool = scene.my_tool

        layout.prop(mytool, "colmap_path")
        layout.operator("my.load_colmap")
        layout.operator("my.debug")
        layout.separator()

class BoundingPanel(NeuralangeloCustomPanel, bpy.types.Panel):
    bl_parent_id  = "panel_main"
    bl_label = "Define Bounding Region"

    def draw(self, context):
        scene = context.scene
        layout = self.layout
        mytool = scene.my_tool
        
        row = layout.row()
        row.alignment = 'CENTER'
        row.label(text="Edit bounding box")

        layout.row().prop(mytool, "my_slider", index=0, slider=True, text='X min')
        layout.row().prop(mytool, "my_slider", index=1, slider=True, text='X max')
        layout.row().prop(mytool, "my_slider", index=2, slider=True, text='Y min')
        layout.row().prop(mytool, "my_slider", index=3, slider=True, text='Y max')
        layout.row().prop(mytool, "my_slider", index=4, slider=True, text='Z min')
        layout.row().prop(mytool, "my_slider", index=5, slider=True, text='Z max')
        layout.separator()

        layout.operator("my.crop")
        layout.operator("my.hide_show_box")
        layout.operator("my.add_bound_sphere")

# ------------------------------------------------------------------------
#    Registration
# ------------------------------------------------------------------------

classes = (
    MyProperties,
    OT_LoadCOLMAP,
    OT_Debug,
    MainPanel,
    LoadingPanel,
    BoundingPanel,
    Crop,
    BoundSphere,
    HideShowBox
)


def register():
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)

    bpy.types.Scene.my_tool = PointerProperty(type=MyProperties)
    bpy.app.handlers.depsgraph_update_post.append(update_cropping_plane)


def unregister():
    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)
    del bpy.types.Scene.my_tool
    bpy.app.handlers.depsgraph_update_post.remove(update_cropping_plane)


if __name__ == "__main__":
    register()

