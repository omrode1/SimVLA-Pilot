from setuptools import find_packages, setup

package_name = 'vla_perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='omrode',
    maintainer_email='omrode.34@gmail.com',
    description='VLA Project Perception Node using YOLOv8',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'perception_node = vla_perception.perception_node:main'
        ],
    },
)
