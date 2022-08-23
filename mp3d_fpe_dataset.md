# MP3D-FPE Dataset: Sequential 360-image dataset for floor plan estimation

The MP3D-FPE dataset, presented here, provides a convenient set of indoor scenes captured as a continuous sequences of images in equirectangular projection ($360\text{\textdegree}$ panorama images). This dataset mimics a hand-held camera going through several rooms in a scene, providing 360-images, registered by its corresponded camera position, which allows us to explore new ideas about indoor geometry estimation with multi-view setting using 360-images particularly.

For reference purposes, the following table presets a comparison between the MP3D-FPE with the available datasets for room layout and floor-plan estimation using 360-images and point cloud data. 

```table```


The MP3D-FPE dataset was collected using the Minos simulator [x], and the Real-world dataset MatterPort 3D[x]. Every scene in the MP3D-FPE presents, RGB images (360-images), Depth map, registered camera poses, an estimated layouts using the HorizonNet [x] pretrained on MP3D as described "here". The data in MP3D-FPE es sorted as follows:

```organization tree```

For downloading this dataset please send us an e-mail to enrique.solarte.pardo@gmail.com or nthu.vslab@gmail.com. For quick experiments, you can download our pilot scene by running ```download_mp3d_fpe.py```

Every scene in MP3D-FPE content a **VO-*** directory, which contents the information related to a path sequence where the camera positions were estimated using OpenVslam [x]. Note that this estimation is defined up to an unknown scale (visual odometry scale) due to a monocular nature of our setting. All the information stored inside **VO-*** refers to every estimated keyframe spawn by the VSLAM system. 


