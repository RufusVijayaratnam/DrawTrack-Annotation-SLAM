import Mapping
import importlib
importlib.reload(Mapping)

render_path = "/mtn/c/Users/Rufus Vijayaratnam/Driverless/Blender/Resources/Renders/"
left_vid = render_path + "track8-left.avi"
right_vid = render_path + "track8-right.avi"

slam = Mapping.Mapper(left_vid, right_vid)
slam.begin()