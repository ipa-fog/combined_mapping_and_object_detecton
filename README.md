# combined_mapping_and_object_detecton
Package to combine 3D Mapping (SLAM) and 2D Object Detection (Deep Learning)

Image Data (RGB) from a mobile Robot gets over WIFI into a external pc where a CNN detect Objects. 

The Robot receives the Bounding Boxes and Object Names from the CNN.

Based on the corresponding Depth Image of Robot, the Point Cloud of the Robot gets classified into the Objects.

Using a SLAM-Algorithm to create a 3D-Map which includes Geometric and Object informations.




