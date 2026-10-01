from setuptools import setup
setup(name="factory_mapping_preview",version="0.1.0",packages=["factory_mapping_preview"],data_files=[("share/ament_index/resource_index/packages",["resource/factory_mapping_preview"]),("share/factory_mapping_preview",["package.xml"])],entry_points={"console_scripts":['preview = factory_mapping_preview.node:main']})
