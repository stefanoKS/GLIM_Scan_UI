"""Optional sensor + monitoring launch. Do not run alongside managed UI processes."""
from launch import LaunchDescription
from launch_ros.actions import Node
from factory_mapping.config import ROOT, load
from factory_mapping.commands import driver

def generate_launch_description():
    config=load(); args=driver(ROOT,config)[4:]
    return LaunchDescription([
        Node(package='livox_ros_driver2',executable='livox_ros_driver2_node',arguments=args,output='screen'),
        Node(package='factory_mapping_monitor',executable='monitor',output='screen'),
        Node(package='factory_mapping_preview',executable='preview',output='screen')
    ])
