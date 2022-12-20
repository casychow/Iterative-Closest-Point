# Iterative Closest Point (ICP)

Iterative Closest Point is a registration algorithm that minimizes the distance between corresponding cloud points so that a source and target cloud point may converge. This ICP algorithm permits cloud points of different lengths to converge. As such, there will be no perfect point-to-point correspondence matches between the two point clouds as the source point cloud will only want to approach its nearest neighbor.

## Part A - ICP on Demo Point Cloud Data
The cloud points used in this algorithm are the default cloud points taken from the Python Open3D library which can be called using `open3d.data.DemoICPPointClouds()` after importing the necessary libraries.

The algorithm was made with open3d's `KDTreeFlan` class.

The following figure shows two point clouds converging after 21 iterations and a final minimized cost function of 7.42 with a cost difference of 0.000001.

![](https://i.imgur.com/lrEcu39.png)




## Part B - ICP on Kitti Point Cloud Data
The cloud points used in this algorithm are the default cloud points taken from the Python Open3D library which can be called using `open3d.io.read_point_cloud('kitti_frame1.pcd')` and `open3d.io.read_point_cloud('kitti_frame2.pcd')` after importing the necessary libraries.

Unlike Part A, Part B's point clouds do not appear to align properly. This is because the point clouds in this part are not densely packed as the point clouds in the first part. As a result, the ICP algorithm experiences more difficulty in determining the correct corresponding nearest neighbor.

The following figure shows the two Kitti point clouds converging after 44 iterations to obtain a cost difference less than 0.001.

![](https://i.imgur.com/wPJHack.png)
