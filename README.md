# Unscented Kalman Filter Project Starter Code
Self-Driving Car Engineer Nanodegree Program

In this project utilize an Unscented Kalman Filter to estimate the state of a moving object of interest with noisy lidar and radar measurements. Passing the project requires obtaining RMSE values that are lower that the tolerance outlined in the project rubric. 

This project involves the Term 2 Simulator which can be downloaded [here](https://github.com/udacity/self-driving-car-sim/releases)

This repository includes two files that can be used to set up and intall [uWebSocketIO](https://github.com/uWebSockets/uWebSockets) for either Linux or Mac systems. For windows you can use either Docker, VMware, or even [Windows 10 Bash on Ubuntu](https://www.howtogeek.com/249966/how-to-install-and-use-the-linux-bash-shell-on-windows-10/) to install uWebSocketIO. Please see [this concept in the classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/16cf4a78-4fc7-49e1-8621-3450ca938b77) for the required version and installation scripts.

Once the install for uWebSocketIO is complete, the main program can be built and ran by doing the following from the project top directory.

1. mkdir build
2. cd build
3. cmake ..
4. make
5. ./UnscentedKF

Tips for setting up your environment can be found [here](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/23d376c7-0195-4276-bdf0-e02f1f3c665d)

Here is the main protcol that main.cpp uses for uWebSocketIO in communicating with the simulator.

INPUT: values provided by the simulator to the c++ program

```
["sensor_measurement"] => the measurment that the simulator observed (either lidar or radar)
```

OUTPUT: values provided by the c++ program to the simulator
```
["estimate_x"] <= kalman filter estimated position x
["estimate_y"] <= kalman filter estimated position y
["rmse_x"]
["rmse_y"]
["rmse_vx"]
["rmse_vy"]
```

---

## Other Important Dependencies
* cmake >= 3.5
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./UnscentedKF` Previous versions use i/o from text files.  The current state uses i/o
from the simulator.

## Generating Additional Data

If you'd like to generate your own radar and lidar data, see the
[utilities repo](https://github.com/udacity/CarND-Mercedes-SF-Utilities) for
Matlab scripts that can generate additional data.

# Results

![Track - Unscented Kalman Filter](NIS/track.png)

The red circles are lidar measurements, blue circles are the positions inferred from radar measurements (radius and angle). 
and green triangles are the estimated car's position by the Unscented Kalman Filter.

The process model "constant turn rate and velocity magnitude" (CTRV) is used for Kalman filter's predict step. 
It tracks a state vector of 5 dimensions: x position, y position, velocity, yaw angle, and yaw rate. 

To predict the state for a new measurement the velocity magnitude and yaw rate are assumed to be constant.
However, a random velocity acceleration and yaw acceleration might change the velocity and yaw rate. 
These two accelaration values can be chosen by reasoning and experiment. A car may accelerate with 1.5m/t^2 
and change the yaw rate with 1 rad/t^2. 

To check if the these values are a good choice one may use "normalized information squared" or NIS statistic. 
The NIS values computed from predicted measurements should not exceed the 95% level of significance of the 
[chi-square distribution](http://uregina.ca/~gingrich/appchi.pdf). It is acceptable if some exceed this level,
but the majority should be below. The degree of freedom is 3 for radar measurement and 2 for lidar measurements. 
The threshold would be then for radar measurements 7.82 and 5.99 for lidar measurements.

The chosen acceleration parameters seem to be right, as shown in the picture below.

![NIS Graphs for Lidar and Radar Measurements](NIS/NIS.png)
 
 
 
