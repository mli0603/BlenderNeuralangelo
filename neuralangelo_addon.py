import collections
import json
import os
import shutil
import struct

import bmesh
import bpy
import math
import numpy as np
from bpy.props import (StringProperty,
                       BoolProperty,
                       FloatProperty,
                       FloatVectorProperty,
                       PointerProperty,
                       )
from bpy.types import (Operator,
                       PropertyGroup,
                       )
from mathutils import Matrix
from typing import Union

# ------------------------------------------------------------------------
#    COLMAP code: https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
# ------------------------------------------------------------------------


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


def convert_to_blender_coord(tvec_w2c, qvec_w2c):
    cv2blender = np.array([[1, 0, 0],
                           [0, -1, 0],
                           [0, 0, -1]])
    R = qvec2rotmat(qvec_w2c)
    tvec_blender = -np.dot(R.T, tvec_w2c)
    rotation = np.dot(R.T, cv2blender)
    qvec_blender = rotmat2qvec(rotation)
    return tvec_blender, qvec_blender


# ------------------------------------------------------------------------
#    Borrowed from BlenderProc:
#    https://github.com/DLR-RM/BlenderProc
# ------------------------------------------------------------------------
def set_intrinsics_from_K_matrix(K: Union[np.ndarray, Matrix], image_width: int, image_height: int,
                                 clip_start: float = None, clip_end: float = None):
    """ Set the camera intrinsics via a K matrix.
    The K matrix should have the format:
        [[fx, 0, cx],
         [0, fy, cy],
         [0, 0,  1]]
    This method is based on https://blender.stackexchange.com/a/120063.
    :param K: The 3x3 K matrix.
    :param image_width: The image width in pixels.
    :param image_height: The image height in pixels.
    :param clip_start: Clipping start.
    :param clip_end: Clipping end.
    """

    K = Matrix(K)

    cam = bpy.context.scene.objects['Input Camera'].data

    if abs(K[0][1]) > 1e-7:
        raise ValueError(f"Skew is not supported by blender and therefore "
                         f"not by BlenderProc, set this to zero: {K[0][1]} and recalibrate")

    fx, fy = K[0][0], K[1][1]
    cx, cy = K[0][2], K[1][2]

    # If fx!=fy change pixel aspect ratio
    pixel_aspect_x = pixel_aspect_y = 1
    if fx > fy:
        pixel_aspect_y = fx / fy
    elif fx < fy:
        pixel_aspect_x = fy / fx

    # Compute sensor size in mm and view in px
    pixel_aspect_ratio = pixel_aspect_y / pixel_aspect_x
    view_fac_in_px = get_view_fac_in_px(cam, pixel_aspect_x, pixel_aspect_y, image_width, image_height)
    sensor_size_in_mm = get_sensor_size(cam)

    # Convert focal length in px to focal length in mm
    f_in_mm = fx * sensor_size_in_mm / view_fac_in_px

    # Convert principal point in px to blenders internal format
    shift_x = (cx - (image_width - 1) / 2) / -view_fac_in_px
    shift_y = (cy - (image_height - 1) / 2) / view_fac_in_px * pixel_aspect_ratio

    # Finally set all intrinsics
    set_intrinsics_from_blender_params(f_in_mm, image_width, image_height, clip_start, clip_end, pixel_aspect_x,
                                       pixel_aspect_y, shift_x, shift_y, "MILLIMETERS")


def get_sensor_size(cam: bpy.types.Camera) -> float:
    """ Returns the sensor size in millimeters based on the configured sensor_fit.
    :param cam: The camera object.
    :return: The sensor size in millimeters.
    """
    if cam.sensor_fit == 'VERTICAL':
        sensor_size_in_mm = cam.sensor_height
    else:
        sensor_size_in_mm = cam.sensor_width
    return sensor_size_in_mm


def get_view_fac_in_px(cam: bpy.types.Camera, pixel_aspect_x: float, pixel_aspect_y: float,
                       resolution_x_in_px: int, resolution_y_in_px: int) -> int:
    """ Returns the camera view in pixels.
    :param cam: The camera object.
    :param pixel_aspect_x: The pixel aspect ratio along x.
    :param pixel_aspect_y: The pixel aspect ratio along y.
    :param resolution_x_in_px: The image width in pixels.
    :param resolution_y_in_px: The image height in pixels.
    :return: The camera view in pixels.
    """
    # Determine the sensor fit mode to use
    if cam.sensor_fit == 'AUTO':
        if pixel_aspect_x * resolution_x_in_px >= pixel_aspect_y * resolution_y_in_px:
            sensor_fit = 'HORIZONTAL'
        else:
            sensor_fit = 'VERTICAL'
    else:
        sensor_fit = cam.sensor_fit

    # Based on the sensor fit mode, determine the view in pixels
    pixel_aspect_ratio = pixel_aspect_y / pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px

    return view_fac_in_px


def set_intrinsics_from_blender_params(lens: float = None, image_width: int = None, image_height: int = None,
                                       clip_start: float = None, clip_end: float = None,
                                       pixel_aspect_x: float = None, pixel_aspect_y: float = None, shift_x: int = None,
                                       shift_y: int = None, lens_unit: str = None):
    """ Sets the camera intrinsics using blenders represenation.
    :param lens: Either the focal length in millimeters or the FOV in radians, depending on the given lens_unit.
    :param image_width: The image width in pixels.
    :param image_height: The image height in pixels.
    :param clip_start: Clipping start.
    :param clip_end: Clipping end.
    :param pixel_aspect_x: The pixel aspect ratio along x.
    :param pixel_aspect_y: The pixel aspect ratio along y.
    :param shift_x: The shift in x direction.
    :param shift_y: The shift in y direction.
    :param lens_unit: Either FOV or MILLIMETERS depending on whether the lens is defined as focal length in
                      millimeters or as FOV in radians.
    """

    cam = bpy.context.scene.objects['Input Camera'].data

    if lens_unit is not None:
        cam.lens_unit = lens_unit

    if lens is not None:
        # Set focal length
        if cam.lens_unit == 'MILLIMETERS':
            if lens < 1:
                raise Exception("The focal length is smaller than 1mm which is not allowed in blender: " + str(lens))
            cam.lens = lens
        elif cam.lens_unit == "FOV":
            cam.angle = lens
        else:
            raise Exception("No such lens unit: " + lens_unit)

    # Set resolution
    if image_width is not None:
        bpy.context.scene.render.resolution_x = image_width
    if image_height is not None:
        bpy.context.scene.render.resolution_y = image_height

    # Set clipping
    if clip_start is not None:
        cam.clip_start = clip_start
    if clip_end is not None:
        cam.clip_end = clip_end

    # Set aspect ratio
    if pixel_aspect_x is not None:
        bpy.context.scene.render.pixel_aspect_x = pixel_aspect_x
    if pixel_aspect_y is not None:
        bpy.context.scene.render.pixel_aspect_y = pixel_aspect_y

    # Set shift
    if shift_x is not None:
        cam.shift_x = shift_x
    if shift_y is not None:
        cam.shift_y = shift_y


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
    "category": "Interface"
}

# global variables for easier access
colmap_data = None
old_box_offset = [0, 0, 0, 0, 0, 0]
view_port = None
point_cloud_vertices = None
select_point_index = []
radius = 0
center = (0, 0, 0)


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
    ply_name = 'Point Cloud'
    mesh = bpy.data.meshes.new(name=ply_name)
    mesh.vertices.add(xyzs.shape[0])
    mesh.vertices.foreach_set("co", [a for v in xyzs for a in v])
    obj = bpy.data.objects.new(ply_name, mesh)
    bpy.context.scene.collection.objects.link(obj)


def generate_cropping_planes():
    global point_cloud_vertices

    max_coordinate = np.max(point_cloud_vertices, axis=0)
    min_coordinate = np.min(point_cloud_vertices, axis=0)

    x_min = min_coordinate[0]
    x_max = max_coordinate[0]
    y_min = min_coordinate[1]
    y_max = max_coordinate[1]
    z_min = min_coordinate[2]
    z_max = max_coordinate[2]

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

    msh = bpy.data.meshes.new('Bounding Box')
    msh.from_pydata(verts, [], faces)
    obj = bpy.data.objects.new('Bounding Box', msh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.scene.objects['Bounding Box'].hide_set(True)

    # Add plane text
    text_object_xmin = bpy.data.objects.new("x_min_label", bpy.data.curves.new(type="FONT", name="x_min"))
    text_object_xmin.data.body = "x min"
    text_object_xmin.data.size *= 2
    bpy.context.scene.collection.objects.link(text_object_xmin)
    bpy.context.scene.objects['x_min_label'].hide_set(True)

    text_object_xmax = bpy.data.objects.new("x_max_label", bpy.data.curves.new(type="FONT", name="x_max"))
    text_object_xmax.data.body = "x max"
    text_object_xmax.data.size *= 2
    bpy.context.scene.collection.objects.link(text_object_xmax)
    bpy.context.scene.objects['x_max_label'].hide_set(True)

    text_object_ymin = bpy.data.objects.new("y_min_label", bpy.data.curves.new(type="FONT", name="y_min"))
    text_object_ymin.data.body = "y min"
    text_object_ymin.data.size *= 2
    bpy.context.scene.collection.objects.link(text_object_ymin)
    bpy.context.scene.objects['y_min_label'].hide_set(True)

    text_object_ymax = bpy.data.objects.new("y_max_label", bpy.data.curves.new(type="FONT", name="y_max"))
    text_object_ymax.data.body = "y max"
    text_object_ymax.data.size *= 2
    bpy.context.scene.collection.objects.link(text_object_ymax)
    bpy.context.scene.objects['y_max_label'].hide_set(True)

    text_object_zmin = bpy.data.objects.new("z_min_label", bpy.data.curves.new(type="FONT", name="z_min"))
    text_object_zmin.data.body = "z min"
    text_object_zmin.data.size *= 2
    bpy.context.scene.collection.objects.link(text_object_zmin)
    bpy.context.scene.objects['z_min_label'].hide_set(True)

    text_object_zmax = bpy.data.objects.new("z_max_label", bpy.data.curves.new(type="FONT", name="z_max"))
    text_object_zmax.data.body = "z max"
    text_object_zmax.data.size *= 2
    bpy.context.scene.collection.objects.link(text_object_zmax)
    bpy.context.scene.objects['z_max_label'].hide_set(True)

    text_object_xmin.rotation_euler = (-math.radians(90), -math.radians(90), math.radians(90))
    text_object_xmax.rotation_euler = (-math.radians(90), -math.radians(90), -math.radians(90))

    text_object_ymin.rotation_euler = (-math.radians(90), 0, math.radians(180))
    text_object_ymax.rotation_euler = (-math.radians(90), 0, 0)

    text_object_zmin.rotation_euler = (-math.radians(180), 0, 0)
    text_object_zmax.rotation_euler = (0, 0, 0)

    text_object_xmin.location = (x_min - 1, (y_max + y_min) / 2, (z_max + z_min) / 2)
    text_object_xmax.location = (x_max + 0.5, (y_max + y_min) / 2, (z_max + z_min) / 2)

    text_object_ymin.location = ((x_max + x_min) / 2, y_min - 1, (z_max + z_min) / 2)
    text_object_ymax.location = ((x_max + x_min) / 2, y_max + 0.5, (z_max + z_min) / 2)

    text_object_zmin.location = ((x_max + x_min) / 2, (y_max + y_min) / 2, z_min - 1)
    text_object_zmax.location = ((x_max + x_min) / 2, (y_max + y_min) / 2, z_max + 0.5)

    return


def update_cropping_plane(self, context):
    global old_box_offset
    global point_cloud_vertices

    if point_cloud_vertices is None:  # stop if point cloud vertices are not yet loaded
        return

    max_coordinate = np.max(point_cloud_vertices, axis=0)
    min_coordinate = np.min(point_cloud_vertices, axis=0)

    x_min = min_coordinate[0]
    x_max = max_coordinate[0]
    y_min = min_coordinate[1]
    y_max = max_coordinate[1]
    z_min = min_coordinate[2]
    z_max = max_coordinate[2]

    slider = bpy.context.scene.my_tool.box_slider
    crop_plane = bpy.data.objects['Bounding Box']

    x_min_change = -slider[0]
    x_max_change = -slider[1]
    y_min_change = -slider[2]
    y_max_change = -slider[3]
    z_min_change = -slider[4]
    z_max_change = -slider[5]

    if -x_min_change != old_box_offset[0] and x_max + x_max_change < x_min - x_min_change:
        x_min_change = x_min - (x_max + x_max_change)
        slider[0] = old_box_offset[0]

    elif -x_max_change != old_box_offset[1] and x_max + x_max_change < x_min - x_min_change:
        x_max_change = x_min - x_min_change - x_max
        slider[1] = old_box_offset[1]

    elif -y_min_change != old_box_offset[2] and y_max + y_max_change < y_min - y_min_change:
        y_min_change = y_min - (y_max + y_max_change)
        slider[2] = old_box_offset[2]

    elif -y_max_change != old_box_offset[3] and y_max + y_max_change < y_min - y_min_change:
        y_max_change = y_min - y_min_change - y_max
        slider[3] = old_box_offset[3]

    elif -z_min_change != old_box_offset[4] and z_max + z_max_change < z_min - z_min_change:
        z_min_change = z_min - (z_max + z_max_change)
        slider[4] = old_box_offset[4]

    elif -z_max_change != old_box_offset[5] and z_max + z_max_change < z_min - z_min_change:
        z_max_change = z_min - z_min_change - z_max
        slider[5] = old_box_offset[5]

    old_box_offset = [n for n in slider]

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

    text_object_xmin = bpy.data.objects['x_min_label']
    text_object_xmax = bpy.data.objects['x_max_label']
    text_object_ymin = bpy.data.objects['y_min_label']
    text_object_ymax = bpy.data.objects['y_max_label']
    text_object_zmin = bpy.data.objects['z_min_label']
    text_object_zmax = bpy.data.objects['z_max_label']
    # update text location and rotation

    text_width = bpy.context.scene.objects['x_min_label'].dimensions[0] / 2

    text_object_xmin.location = (x_min - x_min_change - 1, (y_max + y_max_change + y_min - y_min_change) / 2,
                                 (z_max + z_max_change + z_min - z_min_change) / 2 - text_width)
    text_object_xmax.location = (x_max + x_max_change + 0.5, (y_max + y_max_change + y_min - y_min_change) / 2,
                                 (z_max + z_max_change + z_min - z_min_change) / 2 - text_width)

    text_object_ymin.location = (
        (x_max + x_max_change + x_min - x_min_change) / 2 + text_width, y_min - y_min_change - 1,
        (z_max + z_max_change + z_min - z_min_change) / 2)
    text_object_ymax.location = (
        (x_max + x_max_change + x_min - x_min_change) / 2 - text_width, y_max + y_max_change + 0.5,
        (z_max + z_max_change + z_min - z_min_change) / 2)

    text_object_zmin.location = (
        (x_max + x_max_change + x_min - x_min_change) / 2 - text_width,
        (y_max + y_max_change + y_min - y_min_change) / 2,
        z_min - z_min_change - 1)
    text_object_zmax.location = (
        (x_max + x_max_change + x_min - x_min_change) / 2 - text_width,
        (y_max + y_max_change + y_min - y_min_change) / 2,
        z_max + z_max_change + 1)


def reset_my_slider_to_default():
    bpy.context.scene.my_tool.box_slider[0] = 0
    bpy.context.scene.my_tool.box_slider[1] = 0
    bpy.context.scene.my_tool.box_slider[2] = 0
    bpy.context.scene.my_tool.box_slider[3] = 0
    bpy.context.scene.my_tool.box_slider[4] = 0
    bpy.context.scene.my_tool.box_slider[5] = 0


def delete_bounding_sphere():
    if 'Bounding Sphere' in bpy.data.objects:
        obj = bpy.context.scene.objects['Bounding Sphere']
        bpy.data.meshes.remove(obj.data, do_unlink=True)


# TODO: can this be cleaned up??
# TODO: when loading, not set to solid mode??
def switch_viewport_to_solid(self, context):
    toggle = context.scene.my_tool.transparency_toggle
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'SOLID'
                    space.shading.show_xray = toggle


def enable_texture_mode():
    # change color mode
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.color_type = 'TEXTURE'


def update_transparency(self, context):
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    alpha = context.scene.my_tool.transparency_slider
                    space.shading.xray_alpha = alpha


def set_keyframe_camera(camera, qvec_w2c, tvec_w2c, idx, inter_frames=1):
    # Set rotation and translation of Camera in each frame

    tvec, qvec = convert_to_blender_coord(tvec_w2c, qvec_w2c)

    camera.rotation_quaternion = qvec
    camera.location = tvec

    camera.keyframe_insert(data_path='location', frame=idx * inter_frames)
    camera.keyframe_insert(data_path='rotation_quaternion', frame=idx * inter_frames)


def set_keyframe_image(idx, plane, inter_frames=1):
    # Set vertices of image plane in each frame
    bpy.context.view_layer.update()

    # Set image texture of image plane in each frame
    material = plane.material_slots[0].material
    texture = material.node_tree.nodes.get("Image Texture")
    texture.image_user.frame_offset = idx - 1
    texture.image_user.keyframe_insert(data_path="frame_offset", frame=idx * inter_frames)


def select_all_vert(obj_name):
    if obj_name in bpy.data.objects:
        obj = bpy.context.scene.objects[obj_name]
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(type="VERT")
        bpy.ops.mesh.select_all(action='SELECT')


def generate_camera_plane(camera, image_width, image_height, intrinsic_matrix):
    if 'Image Plane' in bpy.data.objects:
        obj = bpy.context.scene.objects['Image Plane']
        bpy.data.meshes.remove(obj.data, do_unlink=True)

    bpy.context.view_layer.update()

    # create a plane with 4 corners
    verts = camera.data.view_frame()
    faces = [[0, 1, 2, 3]]
    msh = bpy.data.meshes.new('Image Plane')
    msh.from_pydata(verts, [], faces)
    obj = bpy.data.objects.new('Image Plane', msh)
    bpy.context.scene.collection.objects.link(obj)

    plane = bpy.context.scene.objects['Image Plane']
    bpy.context.view_layer.objects.active = plane
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0)

    # change each uv vertex
    bm = bmesh.from_edit_mesh(plane.data)
    uv_layer = bm.loops.layers.uv.active
    for idx, v in enumerate(bm.verts):
        for l in v.link_loops:
            uv_data = l[uv_layer]
            if idx == 0:
                uv_data.uv[0] = 0.0
                uv_data.uv[1] = 0.0
            elif idx == 1:
                uv_data.uv[0] = 0.0
                uv_data.uv[1] = 1.0
            elif idx == 2:
                uv_data.uv[0] = 1.0
                uv_data.uv[1] = 1.0
            elif idx == 3:
                uv_data.uv[0] = 1.0
                uv_data.uv[1] = 0.0
            break

    bpy.ops.object.mode_set(mode='OBJECT')

    plane.parent = camera
    # set plane vertex location
    camera_vert_origin = camera.data.view_frame()
    corners = np.array([
        [0, 0, 1],
        [0, image_height, 1],
        [image_width, image_height, 1],
        [image_width, 0, 1]
    ])
    corners_3D = corners @ (np.linalg.inv(intrinsic_matrix).transpose(-1, -2))
    for vert, corner in zip(camera_vert_origin, corners_3D):
        vert[0] = corner[0]
        vert[1] = corner[1]
        vert[2] = -1.0  # blender coord

    for i in range(4):
        plane.data.vertices[i].co = camera_vert_origin[i]


def copy_all_images(blender_img_path, image_folder_path, image_names, sort_image_id):
    file_names = []
    for idx, sorted_idx in enumerate(sort_image_id):
        shutil.copyfile(image_folder_path + image_names[sorted_idx],
                        blender_img_path + '%05d.' % (idx + 1) + image_names[sorted_idx].split('.')[-1])
        file_names.append('%05d.' % (idx + 1) + image_names[sorted_idx].split('.')[-1])
    return file_names


def generate_camera_plane_texture(image_sequence):
    plane = bpy.context.scene.objects['Image Plane']
    if 'Image Material' not in bpy.data.materials:
        material = bpy.data.materials.new(name="Image Material")
    else:
        material = bpy.data.materials["Image Material"]

    if len(plane.material_slots) == 0:
        plane.data.materials.append(material)

    material = plane.active_material
    material.use_nodes = True

    image_texture = material.node_tree.nodes.new(type='ShaderNodeTexImage')
    principled_bsdf = material.node_tree.nodes.get('Principled BSDF')
    material.node_tree.links.new(image_texture.outputs['Color'], principled_bsdf.inputs['Base Color'])

    image_texture.image = image_sequence
    image_texture.image_user.use_cyclic = True
    image_texture.image_user.use_auto_refresh = True
    image_texture.image_user.frame_duration = 1
    image_texture.image_user.frame_start = 0
    image_texture.image_user.frame_offset = 0


def load_camera(colmap_data, context):
    if 'Input Camera' in bpy.data.cameras:
        camera = bpy.data.cameras['Input Camera']
        bpy.data.cameras.remove(camera)

    if 'Image Material' in bpy.data.materials:
        material = bpy.data.materials['Image Material']
        bpy.data.materials.remove(material, do_unlink=True)

    # Load colmap data
    intrinsic_param = np.array([camera.params for camera in colmap_data['cameras'].values()])
    intrinsic_matrix = np.array([[intrinsic_param[0][0], 0, intrinsic_param[0][2]],
                                 [0, intrinsic_param[0][1], intrinsic_param[0][3]],
                                 [0, 0, 1]])  # TODO: only supports single camera for now

    image_width = np.array([camera.width for camera in colmap_data['cameras'].values()])
    image_height = np.array([camera.height for camera in colmap_data['cameras'].values()])
    image_quaternion = np.stack([img.qvec for img in colmap_data['images'].values()])
    image_translation = np.stack([img.tvec for img in colmap_data['images'].values()])
    camera_id = np.stack([img.camera_id for img in colmap_data['images'].values()]) - 1  # make it zero-indexed
    image_names = np.stack([img.name for img in colmap_data['images'].values()])
    num_image = image_names.shape[0]

    # set start and end frame
    context.scene.frame_start = 1
    context.scene.frame_end = num_image

    # Load image file
    sort_image_id = np.argsort(image_names)
    image_folder_path = bpy.path.abspath(bpy.context.scene.my_tool.colmap_path + 'images/')

    ## make a copy of images to comply with the continuous numbering requirement of sequence
    blender_img_path = bpy.context.scene.my_tool.colmap_path + 'blender_images/'
    if os.path.isdir(blender_img_path):
        if os.listdir(blender_img_path):
            blender_file_names = sorted(os.listdir(blender_img_path))
        else:
            blender_file_names = copy_all_images(blender_img_path, image_folder_path, image_names, sort_image_id)
    else:
        os.mkdir(blender_img_path)
        blender_file_names = copy_all_images(blender_img_path, image_folder_path, image_names, sort_image_id)

    blender_file_names_formatted = [{"name": file_name} for file_name in blender_file_names]
    bpy.ops.image.open(filepath=blender_img_path, directory=blender_img_path, files=blender_file_names_formatted,
                       relative_path=True, show_multiview=False)

    ## sequence named after the first image filename
    image_sequence = bpy.data.images[blender_file_names_formatted[0]['name']]
    image_sequence.source = 'SEQUENCE'

    # Camera initialization
    camera_data = bpy.data.cameras.new(name="Input Camera")
    camera_object = bpy.data.objects.new(name="Input Camera", object_data=camera_data)
    bpy.context.scene.collection.objects.link(camera_object)
    bpy.data.objects['Input Camera'].rotation_mode = 'QUATERNION'

    set_intrinsics_from_K_matrix(intrinsic_matrix, int(image_width[0]),
                                 int(image_height[0]))  # set intrinsic matrix
    camera = bpy.context.scene.objects['Input Camera']

    # Image Plane Setting
    generate_camera_plane(camera, int(image_width[0]), int(image_height[0]), intrinsic_matrix)  # create plane
    generate_camera_plane_texture(image_sequence)

    # Setting Camera & Image Plane frame data
    plane = bpy.context.scene.objects['Image Plane']
    for idx, (i_id, c_id) in enumerate(zip(sort_image_id, camera_id)):
        frame_id = idx + 1  # one-indexed
        set_keyframe_camera(camera, image_quaternion[i_id], image_translation[i_id], frame_id)
        set_keyframe_image(frame_id, plane)

    # enable texture mode to visualize images
    enable_texture_mode()

    # keep point cloud highlighted
    select_all_vert('Point Cloud')

    return


def update_depth(self, context):
    depth_coef = bpy.context.scene.my_tool.imagedepth_slider
    scale_coef = 1 + depth_coef
    camera = bpy.context.scene.objects['Input Camera']
    camera.scale = (scale_coef, scale_coef, scale_coef)


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
    box_slider: FloatVectorProperty(
        name="Plane offset",
        subtype='TRANSLATION',
        description="X_min, X_max ,Y_min ,Y_max ,Z_min ,Z_max",
        size=6,
        min=0,
        max=20,
        default=(0, 0, 0, 0, 0, 0),
        update=update_cropping_plane
    )
    transparency_slider: FloatProperty(
        name="Transparency",
        description="Transparency",
        min=0,
        max=1,
        default=0.1,
        update=update_transparency
    )
    transparency_toggle: BoolProperty(
        name="",
        description="Toggle transparency",
        default=False,
        update=switch_viewport_to_solid
    )
    imagedepth_slider: FloatProperty(
        name="Image Depth",
        description="Depth",
        min=0,
        max=15,
        default=0,
        update=update_depth
    )


# ------------------------------------------------------------------------
#    Operators, i.e, buttons + callback
# ------------------------------------------------------------------------


class LoadCOLMAP(Operator):
    '''
    Load colmap data given file directory, setting up bounding box, set camera parameters and load images
    '''
    bl_label = "Load COLMAP Data"
    bl_idname = "addon.load_colmap"

    @classmethod
    def poll(cls, context):
        return context.scene.my_tool.colmap_path != ''

    def execute(self, context):
        scene = context.scene
        mytool = scene.my_tool

        # remove all objects
        for mesh in bpy.data.meshes:
            bpy.data.meshes.remove(mesh, do_unlink=True)
        for camera in bpy.data.cameras:
            bpy.data.cameras.remove(camera)
        for light in bpy.data.lights:
            bpy.data.lights.remove(light)
        for material in bpy.data.materials:
            bpy.data.materials.remove(material, do_unlink=True)
        for image in bpy.data.images:
            bpy.data.images.remove(image, do_unlink=True)
        for curve in bpy.data.curves:
            bpy.data.curves.remove(curve, do_unlink=True)

        # load data
        cameras, images, points3D = read_model(bpy.path.abspath(mytool.colmap_path + 'sparse/'), ext='.bin')
        display_pointcloud(points3D)

        global colmap_data, point_cloud_vertices

        colmap_data = {}
        colmap_data['cameras'] = cameras
        colmap_data['images'] = images
        colmap_data['points3D'] = points3D

        point_cloud_vertices = np.stack([point.xyz for point in points3D.values()])

        # generate bounding boxes for cropping
        generate_cropping_planes()
        reset_my_slider_to_default()

        # load camera info
        load_camera(colmap_data, context)

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
    bl_idname = "addon.crop"

    @classmethod
    def poll(cls, context):
        global point_cloud_vertices
        return point_cloud_vertices is not None

    def execute(self, context):
        if 'Point Cloud' in bpy.data.objects:
            obj = bpy.context.scene.objects['Point Cloud']
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode='OBJECT')

        global point_cloud_vertices
        global select_point_index

        box_verts = np.array([v.co for v in bpy.data.objects['Bounding Box'].data.vertices])

        max_coordinate = np.max(box_verts, axis=0)
        min_coordinate = np.min(box_verts, axis=0)

        x_min = min_coordinate[0]
        x_max = max_coordinate[0]
        y_min = min_coordinate[1]
        y_max = max_coordinate[1]
        z_min = min_coordinate[2]
        z_max = max_coordinate[2]

        # initialization
        mesh = bpy.data.objects['Point Cloud'].data
        mesh.vertices.foreach_set("hide", [True] * len(mesh.vertices))
        select_point_index = np.where((point_cloud_vertices[:, 0] >= x_min) &
                                      (point_cloud_vertices[:, 0] <= x_max) &
                                      (point_cloud_vertices[:, 1] >= y_min) &
                                      (point_cloud_vertices[:, 1] <= y_max) &
                                      (point_cloud_vertices[:, 2] >= z_min) &
                                      (point_cloud_vertices[:, 2] <= z_max))

        for index in select_point_index[0]:
            bpy.data.objects['Point Cloud'].data.vertices[index].hide = False

        select_all_vert('Point Cloud')

        return {'FINISHED'}


class BoundSphere(Operator):
    '''
    crop points outside the bounding box
    '''

    bl_label = "Create Bounding Sphere"
    bl_idname = "addon.add_bound_sphere"

    @classmethod
    def poll(cls, context):
        global select_point_index
        if select_point_index:
            return True
        else:
            return False

    def execute(self, context):
        global point_cloud_vertices
        global select_point_index
        global radius
        global center

        delete_bounding_sphere()

        unhide_verts = point_cloud_vertices[select_point_index]

        max_coordinate = np.max(unhide_verts, axis=0)
        min_coordinate = np.min(unhide_verts, axis=0)

        x_min = min_coordinate[0]
        x_max = max_coordinate[0]
        y_min = min_coordinate[1]
        y_max = max_coordinate[1]
        z_min = min_coordinate[2]
        z_max = max_coordinate[2]

        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        center_z = (z_min + z_max) / 2

        radius = np.max(np.sqrt((unhide_verts[:, 0] - center_x) ** 2 + (unhide_verts[:, 1] - center_y) ** 2 + (
                unhide_verts[:, 2] - center_z) ** 2))
        center = (center_x, center_y, center_z)

        num_segments = 128
        sphere_verts = []
        sphere_faces = []

        for i in range(num_segments):
            theta1 = i * 2 * np.pi / num_segments
            z = radius * np.sin(theta1)
            xy = radius * np.cos(theta1)
            for j in range(num_segments):
                theta2 = j * 2 * np.pi / num_segments
                x = xy * np.sin(theta2)
                y = xy * np.cos(theta2)
                sphere_verts.append([center[0] + x, center[1] + y, center[2] + z])

        for i in range(num_segments - 1):
            for j in range(num_segments):
                idx1 = i * num_segments + j
                idx2 = (i + 1) * num_segments + j
                idx3 = (i + 1) * num_segments + (j + 1) % num_segments
                idx4 = i * num_segments + (j + 1) % num_segments
                sphere_faces.append([idx1, idx2, idx3])
                sphere_faces.append([idx1, idx3, idx4])

        sphere_mesh = bpy.data.meshes.new('Bounding Sphere')
        sphere_mesh.from_pydata(sphere_verts, [], sphere_faces)
        sphere_mesh.update()

        sphere_obj = bpy.data.objects.new("Bounding Sphere", sphere_mesh)
        bpy.context.scene.collection.objects.link(sphere_obj)

        select_all_vert('Point Cloud')

        return {'FINISHED'}


class HideShowBox(Operator):
    bl_label = "Hide/Show Bounding Box"
    bl_idname = "addon.hide_show_box"

    @classmethod
    def poll(cls, context):
        return point_cloud_vertices is not None

    def execute(self, context):
        status = bpy.context.scene.objects['Bounding Box'].hide_get()
        bpy.context.scene.objects['Bounding Box'].hide_set(not status)
        bpy.context.scene.objects['x_max_label'].hide_set(not status)
        bpy.context.scene.objects['x_min_label'].hide_set(not status)
        bpy.context.scene.objects['y_max_label'].hide_set(not status)
        bpy.context.scene.objects['y_min_label'].hide_set(not status)
        bpy.context.scene.objects['z_max_label'].hide_set(not status)
        bpy.context.scene.objects['z_min_label'].hide_set(not status)
        return {'FINISHED'}


class HideShowSphere(Operator):
    bl_label = "Hide/Show Bounding Sphere"
    bl_idname = "addon.hide_show_sphere"

    @classmethod
    def poll(cls, context):
        return 'Bounding Sphere' in context.scene.collection.objects

    def execute(self, context):
        status = bpy.context.scene.objects['Bounding Sphere'].hide_get()
        bpy.context.scene.objects['Bounding Sphere'].hide_set(not status)
        return {'FINISHED'}


class HideShowCroppedPoints(Operator):
    bl_label = "Hide/Show Cropped Points"
    bl_idname = "addon.hide_show_cropped"

    @classmethod
    def poll(cls, context):
        global select_point_index
        if select_point_index:
            return True
        else:
            return False

    def execute(self, context):
        if 'Point Cloud' in bpy.data.objects:
            obj = bpy.context.scene.objects['Point Cloud']
            bpy.context.view_layer.objects.active = obj
            if obj.mode == 'EDIT':
                bpy.ops.object.mode_set(mode='OBJECT')
            else:
                bpy.ops.object.mode_set(mode='EDIT')
            return {'FINISHED'}


class ExportSceneParameters(Operator):
    bl_label = "Export Scene Parameters"
    bl_idname = "addon.export_scene_param"

    @classmethod
    def poll(cls, context):
        return 'Bounding Sphere' in context.scene.collection.objects

    def execute(self, context):
        global radius, center, colmap_data
        intrinsic_param = np.array([camera.params for camera in colmap_data['cameras'].values()])
        fl_x = intrinsic_param[0][0]  # TODO: only supports single camera for now
        fl_y = intrinsic_param[0][1]
        cx = intrinsic_param[0][2]
        cy = intrinsic_param[0][3]
        image_width = np.array([camera.width for camera in colmap_data['cameras'].values()])
        image_height = np.array([camera.height for camera in colmap_data['cameras'].values()])
        w = image_width[0]
        h = image_height[0]

        angle_x = math.atan(w / (fl_x * 2)) * 2
        angle_y = math.atan(h / (fl_y * 2)) * 2

        out = {
            "camera_angle_x": angle_x,
            "camera_angle_y": angle_y,
            "fl_x": fl_x,
            "fl_y": fl_y,
            "k1": 0.0,  # take undistorted images only
            "k2": 0.0,
            "k3": 0.0,
            "k4": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "is_fisheye": False,  # TODO: not supporting fish eye camera
            "cx": cx,
            "cy": cy,
            "w": int(w),
            "h": int(h),
            "aabb_scale": 1.0,  # unit sphere
            "scene_center": center,
            "scene_radius": radius,
            "frames": []
        }

        flip_mat = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])

        path = bpy.context.scene.my_tool.colmap_path

        # read poses
        for img in colmap_data['images'].values():
            rotation = qvec2rotmat(img.qvec)
            translation = (img.tvec - center) / radius
            translation = translation.reshape(3, 1)
            w2c = np.concatenate([rotation, translation], 1)
            w2c = np.concatenate([w2c, np.array([0, 0, 0, 1])[None]], 0)
            c2w = np.linalg.inv(w2c)
            c2w = c2w @ flip_mat  # convert to GL convention used in NGP

            frame = {"file_path": path + '/images/' + img.name, "transform_matrix": c2w.tolist()}
            # print(frame)
            out["frames"].append(frame)

        file_path = bpy.path.abspath(bpy.context.scene.my_tool.colmap_path + 'transforms.json')
        with open(file_path, "w") as outputfile:
            json.dump(out, outputfile, indent=2)
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        file_path = bpy.path.abspath(bpy.context.scene.my_tool.colmap_path + 'transforms.json')
        layout.row().label(text="Parameters exported to " + file_path)


class HideShowImagePlane(Operator):
    bl_label = "Hide/Show Image Plane"
    bl_idname = "addon.hide_show_cam_plane"

    @classmethod
    def poll(cls, context):
        return 'Image Plane' in context.scene.collection.objects and colmap_data is not None

    def execute(self, context):
        status = bpy.context.scene.objects['Image Plane'].hide_get()
        bpy.context.scene.objects['Image Plane'].hide_set(not status)
        return {'FINISHED'}


class HighlightPointcloud(Operator):
    bl_label = "Highlight Pointcloud"
    bl_idname = "addon.highlight_pointcloud"

    @classmethod
    def poll(cls, context):
        # do not enable when point cloud is not loaded
        return 'Point Cloud' in context.scene.collection.objects and colmap_data is not None

    def execute(self, context):
        select_all_vert('Point Cloud')
        return {'FINISHED'}


# ------------------------------------------------------------------------
#    Panel
# ------------------------------------------------------------------------

class NeuralangeloCustomPanel(bpy.types.Panel):
    bl_category = "Neuralangelo"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"


class MainPanel(NeuralangeloCustomPanel, bpy.types.Panel):
    bl_idname = "BN_PT_main"
    bl_label = "Neuralangelo Addon"

    def draw(self, context):
        scene = context.scene
        layout = self.layout
        mytool = scene.my_tool


class LoadingPanel(NeuralangeloCustomPanel, bpy.types.Panel):
    bl_parent_id = "BN_PT_main"
    bl_idname = "BN_PT_loading"
    bl_label = "Load Data"

    def draw(self, context):
        scene = context.scene
        layout = self.layout
        mytool = scene.my_tool

        layout.prop(mytool, "colmap_path")
        layout.operator("addon.load_colmap")
        layout.separator()


class InspectionPanel(NeuralangeloCustomPanel, bpy.types.Panel):
    bl_parent_id = "BN_PT_main"
    bl_idname = "BN_PT_inspection"
    bl_label = "Inspect COLMAP Results"

    def draw(self, context):
        scene = context.scene
        layout = self.layout
        mytool = scene.my_tool

        # visualization
        box = layout.box()
        row = box.row()
        row.alignment = 'CENTER'
        row.label(text="Visualization")

        row = box.row(align=True)
        row.prop(mytool, "transparency_toggle")
        sub = row.row()
        sub.prop(mytool, "transparency_slider", slider=True, text='Transparency of Objects')
        sub.enabled = mytool.transparency_toggle

        box.row().operator("addon.hide_show_cam_plane")
        box.row().operator("addon.hide_show_box")
        box.row().operator("addon.highlight_pointcloud")
        row = box.row()
        row.alignment = 'CENTER'
        row.label(text="Slide Image Along View")
        box.row().prop(mytool, "imagedepth_slider", slider=True, text='Image plane depth')


class BoundingPanel(NeuralangeloCustomPanel, bpy.types.Panel):
    bl_parent_id = "BN_PT_main"
    bl_idname = "BN_PT_bounding"
    bl_label = "Define Bounding Region"

    def draw(self, context):
        scene = context.scene
        layout = self.layout
        mytool = scene.my_tool

        # bounding box
        box = layout.box()
        row = box.row()
        row.alignment = 'CENTER'
        row.label(text="Edit Bounding Box")

        x_row = box.row()
        x_row.prop(mytool, "box_slider", index=0, slider=True, text='X min')
        x_row.prop(mytool, "box_slider", index=1, slider=True, text='X max')

        y_row = box.row()
        y_row.prop(mytool, "box_slider", index=2, slider=True, text='Y min')
        y_row.prop(mytool, "box_slider", index=3, slider=True, text='Y max')

        z_row = box.row()
        z_row.prop(mytool, "box_slider", index=4, slider=True, text='Z min')
        z_row.prop(mytool, "box_slider", index=5, slider=True, text='Z max')

        box.separator()
        row = box.row()
        row.operator("addon.crop")
        row.operator("addon.hide_show_cropped")

        layout.separator()

        # bounding sphere
        box = layout.box()
        row = box.row()
        row.alignment = 'CENTER'
        row.label(text="Create Bounding Sphere")
        box.row().operator("addon.add_bound_sphere")
        box.row().operator("addon.hide_show_sphere")
        box.row().operator('addon.export_scene_param')


# ------------------------------------------------------------------------
#    Registration
# ------------------------------------------------------------------------

classes = (
    MyProperties,
    MainPanel,
    LoadingPanel,
    InspectionPanel,
    BoundingPanel,
    LoadCOLMAP,
    Crop,
    BoundSphere,
    HideShowBox,
    HideShowSphere,
    HideShowCroppedPoints,
    HideShowImagePlane,
    ExportSceneParameters,
    HighlightPointcloud
)


def register():
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)

    bpy.types.Scene.my_tool = PointerProperty(type=MyProperties)


def unregister():
    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)
    del bpy.types.Scene.my_tool


if __name__ == "__main__":
    register()
