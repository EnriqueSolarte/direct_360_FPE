# 360-DFPE: Leveraging Monocular 360-Layouts for Direct Floor Plan Estimation

This work was accepted to the [IEEE Robotics and Automation Letters RA-L](https://ieeexplore.ieee.org/document/9772341)" 



### Introduction
For a quick overview please vist our [website project](https://enriquesolarte.github.io/robust_360_8pa/) and watch our folwing video demo.
[![](https://i.imgur.com/7X3lGKH.png)](https://drive.google.com/file/d/1-ifw3MlV9aCktkXOX8P230gXqofl3QKc/view?usp=sharing)



---
### News

* **06/17/2022**: Code released 

* **06/17/2022**: Pilot scenes released

* *mp3d_fpe dataset soon*
---

### Description

This is the implementation of **360-DFPE** for sequential floor plan estimation using only monocular 360-images as input. 

Using this **REPO** you can:

*  Register multiple 360-images with estimated camera poses, assuming an unknown visual odometry scale and missed camera height. This Registration is accomplished by our proposed Scaler Recovery. See Sec III-C in our paper.
*  Identify every room in the floor plan without using image features but layout geoemtries only. See Sec III-D.
*  Estimated and optimze the room-geometries as set of minimum room corners.See Sec III-F.
*  Excecute **360-DFPE** for single and multiple rooms scenes in a sequential and non-sequential manner. 
*  Evaluates corner and room metrics for the floor plan estimates.


---
### Main Requirements 
* python                    3.7.7
* vispy                     0.5.3
* numpy                     1.18.5 
* opencv-python             3.4.3.18
* pandas                    1.0.5 


---

### Dataset
The dataset used in this project is our own collected **MP3D-FPE** dataset, which uses [MINOS](https://minosworld.github.io/) simulator with real-world [MP3D](https://niessner.github.io/Matterport/) data. We carefully simulate a handled-camera walking through different rooms and registering sequence of 360-images frames. 

For convenience, we have prepared a light-pilot-set of scenes which can be downloaded by running ```download_mp3d_fpe.py```. For accessing to the whole dataset, please send us an email at enrique.solarte.pardo@gmail.com, or nthu.vslab@gmail.com.

For more details about our dataset, please see [MP3D-FPE dataset](mp3d_fpe_dataset.md)

### Settings

For convience, we handle all the involved hyperameter in a yaml file. ```e.g .config/config_TUM_VI.yaml```. You can use the following lines for loading this configuration. 



## Acknowledgement
- Thanks to professor [Wei-Chen Chiu](https://walonchiu.github.io/) for his unvaluable advises.
- Credit of this repo is shared with [Chin-Hsuan Wu](https://chinhsuanwu.github.io/).

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

