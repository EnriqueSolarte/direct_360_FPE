# MP3D-FPE Dataset: Sequential 360-image dataset for floor plan estimation

The `MP3D-FPE` dataset is a set of indoor scenes captured as a continuous sequences of images in equirectangular projection (360-FoV panorama images). 
This dataset mimics a hand-held camera going through several rooms in a scene, providing 360-images, registered by its corresponded camera position and depth information, 
allowing to explore new ideas about indoor geometry estimation. For instance, in [Direct_360_FPE RA-L'22](https://github.com/EnriqueSolarte/direct_360_FPE) 
this dataset is used for the task of floor plan estimation using only images with estimated camera poses (no depth maps nor additional priors), nevertheless, 
the dataset can be used for other tasks such as 3D reconstruction, depth estimation, and more.

## Dataset Structure

The MP3D-FPE dataset was collected using the [MINOS simulator](https://minosworld.github.io/), and the Real-world dataset [MatterPort 3D](https://niessner.github.io/Matterport/). 
Every scene in the `MP3D-FPE` presents, RGB images (360-images), depth maps, camera poses, and estimated layouts using the [HorizonNet](https://github.com/sunset1995/HorizonNet). 
The data in `MP3D-FPE` is released in the following structure:

```
-- {SCENE_ID}
    `-- {SCENE_VERSION}
        |-- depth/ # Depth maps
        |-- rgb/ # 360-images
        |-- frm_ref.txt
        |-- label.json
        |-- mvl_labels.json
        |-- pcl.ply
        `-- vo_final # Visual Odometry 
           |-- cam_pose_estimated.csv
           |-- hn_mp3d # HorizonNet Layouts
           `-- keyframe_list.txt
```

## Download the dataset

> [!WARNING]  
> To access to the `MP3D-FPE` dataset, you need to create an login on HuggingFace and accept the terms and conditions described [HERE](https://huggingface.co/datasets/EnriqueSolarte/mp3d_fpe).
> You will need around ~320GB to store the entire dataset. Note that this dataset was created by using [Matterport 3D](https://niessner.github.io/Matterport/) dataset and [MINOS simulator](https://minosworld.github.io/),
> therefore, the data policies and copyrights are under terms and conditions of established by Matterport 3D, and MINOS projects.

```bash
huggingface-cli download EnriqueSolarte/mp3d_fpe --repo-type dataset --local-dir ${LOCAL_DIR}
```

![](https://i.imgur.com/QxyEfdZ.gif)

### Some qualitative results 
![](https://i.imgur.com/32z3q3i.png)
![](https://i.imgur.com/xgvAm4d.png)

## Citation
```bibtex
@article{Solarte2022_DFPE,
  author={Solarte, Bolivar and Liu, Yueh-Cheng and Wu, Chin-Hsuan and Tsai, Yi-Hsuan and Sun, Min},
  journal={IEEE Robotics and Automation Letters}, 
  title={360-DFPE: Leveraging Monocular 360-Layouts for Direct Floor Plan Estimation}, 
  year={2022},
  volume={7},
  number={3},
  pages={6503-6510},
  doi={10.1109/LRA.2022.3173730}}
```
