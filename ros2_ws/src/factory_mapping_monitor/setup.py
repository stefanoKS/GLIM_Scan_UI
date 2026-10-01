from setuptools import setup
setup(name="factory_mapping_monitor",version="0.1.0",packages=["factory_mapping_monitor"],data_files=[("share/ament_index/resource_index/packages",["resource/factory_mapping_monitor"]),("share/factory_mapping_monitor",["package.xml"])],entry_points={"console_scripts":['monitor = factory_mapping_monitor.node:main']})
