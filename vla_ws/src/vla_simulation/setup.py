import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'vla_simulation'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='omrode',
    maintainer_email='omrode.34@gmail.com',
    description='VLA Project Simulation Node using PyBullet',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'simulation_node = vla_simulation.simulation_node:main'
        ],
    },
)
