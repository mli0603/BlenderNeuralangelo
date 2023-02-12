# You must set $COLMAP_EXE_PATH to 
# the directory containing the COLMAP executables.
$COLMAP_EXE_PATH/patch_match_stereo \
  --workspace_path . \
  --workspace_format COLMAP \
  --PatchMatchStereo.max_image_size 2000 \
  --PatchMatchStereo.geom_consistency true
$COLMAP_EXE_PATH/stereo_fusion \
  --workspace_path . \
  --workspace_format COLMAP \
  --input_type geometric \
  --output_path ./fused.ply
$COLMAP_EXE_PATH/poisson_mesher \
  --input_path ./fused.ply \
  --output_path ./meshed-poisson.ply
$COLMAP_EXE_PATH/delaunay_mesher \
  --input_path . \
  --input_type dense 
  --output_path ./meshed-delaunay.ply
