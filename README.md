## MMPOSE API

This program uses the mmpose API to detect facial landmarks in a video. We acheive this by taking a video, breaking it down into its frames, running inference on each frame, and then piecing the video back together.

### Installation instructions
1. Setup a new conda env (should already be installed assuming you're using discovery)
    - Follow the instructions here: https://mmpose.readthedocs.io/en/latest/installation.html#prerequisites
2. Also run `mim install "mmpose>=1.1.0"` in the terminal after that.

### Running Script
The only script here is `video_api.py`.

This can be run by running the following:
```Bash
python3 video_api.py <path to video>
```
