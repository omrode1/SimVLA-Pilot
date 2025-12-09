from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    use_gui_arg = DeclareLaunchArgument(
        'use_gui',
        default_value='True',
        description='Whether to run PyBullet in GUI mode'
    )
    
    return LaunchDescription([
        use_gui_arg,
        Node(
            package='vla_simulation',
            executable='simulation_node',
            name='simulation_node',
            output='screen',
            parameters=[{'use_gui': LaunchConfiguration('use_gui')}]
        ),
        Node(
            package='vla_perception',
            executable='perception_node',
            name='perception_node',
            output='screen'
        ),
        Node(
            package='vla_brain',
            executable='llm_node',
            name='llm_node',
            output='screen'
        ),
        Node(
            package='vla_brain',
            executable='planner_node',
            name='planner_node',
            output='screen'
        ),
    ])
