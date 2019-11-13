# Video Scene Segmentaion 

![intro](./intro_pic.png)

## content-based-method-for-scene-transition-identification

- detect.py: contains methods for video segmentation initiation and corresponding videos creation.
- run_scene_detection.py: contains example to start the scene segmentation process.

- - - -


## Latency analysis w.r.t to business videos (news)

#### segmentation and video preparation

| (format-size):Duration(secs)            | "Processing time(secs)" | "Video creation time(secs)" |
|:---------------------------------------:|:-----------------------:|:---------------------------:|
| (".mp4"-"12.9mb") : 229.65(3.50 mins)   | 18.81                   |66.92                        |
| (".mp4"-"20.6mb") : 368.78(6.06 mins)   | 29.31	                |75.81                        |
| (".mp4"-"24.4mb") : 561.78(9.22 mins)   | 44.75	                |120.48                       |
| (".mp4"-"28.1mb") : 717.59(11.58 mins)  | 57.69 	                |141.28                       |
| (".mp4"-"225.4mb") : 1783.57(29.44 mins)| 144.97	                |604.10                       |

*Author: Sanjyot Zade*
