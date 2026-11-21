from glob import glob

from setuptools import setup

package_name = "go2_sim_tools"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "fake_lidar = go2_sim_tools.fake_lidar:main",
            "sport_api_bridge = go2_sim_tools.sport_api_bridge:main",
            "world_tf = go2_sim_tools.world_tf:main",
            "throw_scene = go2_sim_tools.throw_scene:main",
            "throw_reflex = go2_sim_tools.throw_reflex:main",
        ],
    },
)
