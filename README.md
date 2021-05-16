
# Setup Instuctions

You must have an updated version of [Blender](https://www.blender.org/) installed to render new track layouts.

Open a terminal and enter the following commands:

`git clone https://github.com/RufusVijayaratnam/SLAM.git`

`cd SLAM`

`pip install -r requirements.txt`

**Note:** If on Windows, then PyTorch cannot be install normally. 

- Go to the [PyTorch website](https://pytorch.org/) and choose the relevant options to install PyTorch. 
- CUDA is not necessary but will speed up [YOLOv5](https://github.com/ultralytics/yolov5) training.

# Drawing a new track

In the *SLAM* directory:

`cd 'Draw and Annotate'`

To draw a new track called "example"

`python DrawTrack.py example`

A window will open, you must sequentially click to add points representing the track layout. The green dot represents the starting point.

Press "esc" to finish drawing.

# Rendering

Open Blender Environment called *FSD Environment.blend* located in the Blender directory.

Click **Window > Toggle System Console** to open the console.

Navigate to the scripting tab and select the *ConePlacement.py.001* script and set the following variable on line 15:

`track = "example.txt"`

Run the script.

The track will be imported into the Blender environment.


## Render a video
To render the sequential images for a full video render, select the *RenderVideo.py* script, set the following variable on line 9:

`track = "example.txt"`

Run the script.

Blender will become unresponsive, the console will show the progress of the track render.

## Create a video

- You must have [FFmpeg](https://www.ffmpeg.org/) installed.

In the *SLAM* directory:

`cd Blender/Resources/Renders/video/images`

To create an .mp4 file for each stereo camera:

`ffmpeg -framerate 30 -i example-Right_Cam-Render-%04d.png -pix_fmt yuv420p ../../Videos/example-right.mp4 `

Replace *example* with the name of your track.

Repeat the command but replace *Right_Cam* with *Left_Cam*