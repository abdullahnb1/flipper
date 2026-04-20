To run this package you first need to create a ROS2 Humble workspace.
- Copy this package into the src folder in your workspace
- Go to the root of your workspace and build the package with,
    colcon build --symlink-install --packages-select camera_pkg
- After you build the package you can start publishing camera frames with,
    ros2 run camera_pkg camera_publisher_node
- and receive those frames with
    ros2 run camera_pkg camera_subscriber_node

I wrote the commands from the top of my head, there might be some errors or typos.
Here is a usefull tutorial for ROS2 (if the link doesn't work, just search for the title in YouTube):
- [ROS2 Tutorials - ROS2 Humble For Beginners](https://youtube.com/playlist?list=PLLSegLrePWgJudpPUof4-nVFHGkB62Izy&si=N9_76iKVC4n1-SZIhttps://youtube.com/playlist?list=PLLSegLrePWgJudpPUof4-nVFHGkB62Izy&si=N9_76iKVC4n1-SZI)