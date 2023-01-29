# 360-DFPE: Leveraging Monocular 360-Layouts for Direct Floor Plan Estimation

This project was published in [IEEE Robotics and Automation Letters RA-L](https://ieeexplore.ieee.org/document/9772341)                      
[[ArXiv Paper]](https://arxiv.org/abs/2112.06180)

### Introduction
For a quick overview please visit our [website project](https://enriquesolarte.github.io/360-dfpe/) or watch our video demo depicted below. For further questions, please contact us to enrique.solarte.pardo@gmail.com
[![](https://i.imgur.com/7X3lGKH.png)](https://drive.google.com/file/d/1-ifw3MlV9aCktkXOX8P230gXqofl3QKc/view?usp=sharing)



---
### News

* **19/07/2022**: Code and pilot scenes released 

* **29/01/2023**: Free access to MP3D-FPE.  
---

### Description

This is the implementation of the **360-DFPE**, sequential floor plan estimation using only monocular 360-images as input. 

Using this **REPO**, you can:

*  Excecute **360-DFPE** for single and multiple room scenes in a sequential and non-sequential manner. 
*  Register multiple 360-layouts with estimated camera poses, assuming an unknown visual odometry scale and missed camera height. For convenience, this registration can be executed isolated from the floor plan estimation pipeline. see [Computing metadata](#computing-metadata)
*  Evaluate corner and room metrics for floor plan estimation.

---
### Main Requirements 
* python                    3.7.7
* vispy                     0.5.3
* numpy                     1.18.5 
* opencv-python             3.4.3.18
* pandas                    1.0.5 


---

### Dataset
The dataset used in this project is our own collected **MP3D-FPE** dataset, which simulates a handled-camera walking through different rooms scenes rendered by [MINOS](https://minosworld.github.io/) simulator with [MP3D](https://niessner.github.io/Matterport/) real-world data. Upon this collected data, we annotate floor plan labels, estimated camera poses uisng OpenVSLAM[4], and estimated layout for each key-frame by HorizonNet[3]. Our dataset offers 360-rgb-images, depth maps, ground truth and estimated camera poses, floor plan labels, and estimated layouts.  

<!-- ![](https://i.imgur.com/qd80tlW.gif) -->
![](https://i.imgur.com/QxyEfdZ.gif)

For convenience, we have prepared a light-set of scenes which can be downloaded by running ```download_mp3d_fpe.py```. For accessing to the whole dataset, please send us an email to enrique.solarte.pardo@gmail.com, or nthu.vslab@gmail.com.

<!-- For more details about our dataset, please see [MP3D-FPE dataset](mp3d_fpe_dataset.md)

For running **360-DFPE** using a custom dataset, please see  [Runing 360-DFPE on custom dataset](360_dfpe_in_custom_data.md)
### Settings -->

All hyperameter settings are stored in a yaml file at ```./config/config.yaml```. Addtitionally, for practical purposes, we define a data manager class ```DataManager```, which handles the data in the **MP3D-FPE** dataset, i.e., ground truth information, estimated poses, rgb images, etc. The following is a typical initialization of our system:  

```py
from config import read_config
from data_manager import DataManager
from src import DirectFloorPlanEstimation

config_file = "./config/config.yaml"

cfg = read_config(config_file=config_file)
dt = DataManager(cfg)
fpe = DirectFloorPlanEstimation(dt)
...
fpe.scale_recover.esimate_vo_scale()
...

```
### How to execute 360-DFPE

For executing **360-DFPE**, we have created three main scripts, i.e., ```main_eval_scene.py```, ```main_eval_list_scenes.py```, ```main_eval_non_seq_approach.py```. 

##### Running on a single scene (Single or Multiple rooms)
```sh
python main_eval_scene.py --scene_name 1LXtFkjw3qL_0 --results ./test/
```

##### Running from a list of scenes
```sh
python main_eval_scenes_list.py --scene_list ./data/scene_list_pilot.txt --results ./test/
```

```sh
python main_eval_non_seq_approach.py --scene_list ./data/scene_list_pilot.txt --results ./test/
```

### Computing metadata

Note that our formulation rely on monocular estimated camera poses, therefore the real scale of the odometry is missed. Additionally, since every layout geometry is estimated from a monocular inference of HorizonNet[3], the layout scale is also missed. For these reasons, we additionally implemented ```main_compute_metadata.py```, which will compute these missed scales along with some additional information such as ground truth rooms, labels, and room-corners, and the missed scales need for 360-layout registration. For convenience, this script runs isolated from the floor plan estimation pipeline. 

```sh
python main_compute_metadata.py --scene_list ./data/scene_list_pilot.txt --results ./test/
```

![](https://i.imgur.com/xgvAm4d.png)


### Some qualitative results 
![](https://i.imgur.com/32z3q3i.png)


## Acknowledgement
- Thanks to professor [Wei-Chen Chiu](https://walonchiu.github.io/) for his unvaluable advises in this project.
- The credits of this repo are shared with [Yueh-Cheng Liu](https://liu115.github.io/).


## Citation
Please cite our paper for any purpose of usage.
```
@ARTICLE{9772341,
  author={Solarte, Bolivar and Liu, Yueh-Cheng and Wu, Chin-Hsuan and Tsai, Yi-Hsuan and Sun, Min},
  journal={IEEE Robotics and Automation Letters}, 
  title={360-DFPE: Leveraging Monocular 360-Layouts for Direct Floor Plan Estimation}, 
  year={2022},
  volume={7},
  number={3},
  pages={6503-6510},
  doi={10.1109/LRA.2022.3173730}}

```
---
### References
[1]: [Chen, Jiacheng, et al. "Floor-sp: Inverse cad for floorplans by sequential room-wise shortest path." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.](https://github.com/woodfrog/floor-sp)

[2]: [Liu, Chen, Jiaye Wu, and Yasutaka Furukawa. "Floornet: A unified framework for floorplan reconstruction from 3d scans." Proceedings of the European conference on computer vision (ECCV). 2018.](https://github.com/art-programmer/FloorNet)

[3]: [Sun, Cheng, et al. "Horizonnet: Learning room layout with 1d representation and pano stretch data augmentation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.](https://sunset1995.github.io/HorizonNet/)

[4]: [Sumikura, Shinya, Mikiya Shibuya, and Ken Sakurada. "OpenVSLAM: A versatile visual SLAM framework." Proceedings of the 27th ACM International Conference on Multimedia. 2019.](https://github.com/fabianschenk/openvslam-1#:~:text=OpenVSLAM%20is%20a%20monocular%2C%20stereo,based%20on%20the%20prebuilt%20maps.)
