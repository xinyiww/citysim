# CitySim-ROS-Interface

[Updated: Apr. 11, 2023]

**CitySim-ROS-Interface** is designed to be a plug-in python interface that works with prediction modules developed in prediction_ct_vel repo. Its function includes
1. Importing Citysim trajectory dataset and transform it to bags.
2. data visualization
3. ADE, FDE Evaluation 

# demo here


## Prerequisites
- Python 3.8
- [Carla 0.9.11](https://github.com/honda-research-institute/carla-setup/tree/0.9.11)
- [ROS Noetic](http://wiki.ros.org/noetic/Installation)


## Download Carla Simulator and setup the workspace

Use Carla version 0.9.11 with Python 3.8 kernel here.

1. Download the version 0.9.11 release from https://github.com/carla-simulator/carla/releases/tag/0.9.11.

2. Setup environment variables

Include CARLA Python API to the Python path:

```
export CARLA_ROOT=/path/to/your/carla/installation
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/agents
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg
```

Additional path might need adding, listed for reference:

```
export PYTHONPATH=$PYTHONPATH:/home/xliu/Documents/carla-setup/catkin_ws/src/carla_setup_src/msgs/ #Adds the path to the Carla ROS message definitions to the PYTHONPATH environment variable.

export UE4_ROOT=/path/to/your/UE4/installation # Sets the UE4_ROOT environment variable to the path where the Unreal Engine is installed.

source /opt/ros/noetic/setup.bash # Sets up the environment for ROS (Robot Operating System).

source /home/xliu/Documents/carla-setup/catkin_ws/devel/setup.bash #Sets up the environment for the Carla ROS bridge.

```

## Run prediction rosnode (detailed procedure in the prediction_ct_vel module). 


Using script
run_eval.sh
run_sim_time.sh 
revise the filname in the shell script 
