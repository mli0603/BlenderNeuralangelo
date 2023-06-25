import bpy
import os 

dir_path = os.path.dirname(os.path.realpath(__file__))

# install
bpy.ops.preferences.addon_install(filepath=os.path.join(dir_path, "neuralangelo_addon.py"))
# enable
bpy.ops.preferences.addon_enable(module='neuralangelo_addon')