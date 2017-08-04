---
# Recurrent Neural Network based Pixel Prediction and In-painting
## Advanced Topics in Machine Learning
### Course at UCL (University College London) by Google DeepMind  
#### COMPGI13 Assignment 2 - Inpainting using Recurrent Neural Networks (LSTM) on MNIST dataset
---
#### Nitish Mutha

##### email: nitish.mutha.16@ucl.ac.uk 
---  

#### Recurrent Neural Network based Pixel Prediction and In-painting demo:  

![demo](https://github.com/NitishMutha/RNN-pixel-prediction/blob/master/data/demo.gif)

For each task I have added a command line API, so you can choose to run any model in either TRAIN or TEST mode. You can also select to run a model with specific configurations. In test mode code will pick models automatically based on arguments passed. For each task following are the API avaliable to execute the code from command line  

#### Dependencies
* Python 3.5.2
* TensorFlow version 1.0  
* sklearn  
* numpy  
* pandas  
* matplotlib  



## Task 1
**Code:** `code/task1.py`  
Note: Additional task1-vannial.py code has been provided with the basic implementation of the LSTM/GRU without any regualarisation.

**Models folders:** (*in test mode code will pick models automatically based on arguments passed)  
1. LSTM 32 units, 1 layer: `models/task1_a_32`  
2. LSTM 64 units, 1 layer: `models/task1_a_64`  
3. LSTM 128 units, 1 layer: `models/task1_a_128`  
4. GRU 32 units, 1 layer: `models/task1_b_32`  
5. GRU 64 units, 1 layer: `models/task1_b_64`  
6. GRU 128 units, 1 layer: `models/task1_b_128`  
7. Stacked LSTM 32 units, 3 layers: `models/task1_c_32`  
8. Stacked GRU 32 units, 3 layers: `models/task1_d_32`  

####API to run the code from command line:  
Command: `python task1.py <arg1> <arg2> <arg3> <arg4>`  
Where,   
arg1 = (str) mode: e.g. `train`, `test`  
arg2 = (str) model: e.g. `lstm`, `gru`  
arg3 = (int) rnn size: e.g. `32` or `64` or `128`  
arg4 = (int) number of layers: e.g. `1` or `3`   

**Example:**
`python task1.py test gru 32 1`

---

## Task 2(a)
**Code:** `code/task2a.py`  

**Models folders:** (Trained with GRU only)  
1. GRU 32 units, 1 layer: `models/task2_gru_32`  
2. GRU 64 units, 1 layer: `models/task2_gru_64`  
3. GRU 128 units, 1 layer: `models/task2_gru_128`    
4. Stacked GRU 32 units, 3 layers: `models/task2_gru_32_3`  

####API to run the code from command line:  
Command: `python task2a.py <arg1> <arg2> <arg3> <arg4>`  
Where,   
arg1 = (str) mode: e.g. `train`, `test`  
arg2 = (str) model: e.g. `gru`  
arg3 = (int) rnn size: e.g. `32` or `64` or `128`  
arg4 = (int) number of layers: e.g. `1` or `3`   

**Example:**
`python task2a.py test gru 32 1`

---

## Task 2(b)
**Code:** `code/task2b.py`  

**Models folders:** (Run on trained GRU model only)  
1. GRU 32 units, 1 layer: `models/task2_gru_32`  
2. GRU 64 units, 1 layer: `models/task2_gru_64`  
3. GRU 128 units, 1 layer: `models/task2_gru_128`    
4. Stacked GRU 32 units, 3 layers: `models/task2_gru_32_3`  

####API to run the code from command line:  
Command: `python task2b.py <arg1> <arg2> <arg3> <arg4> <arg5> <arg6>`  
Where,   
arg1 = (str) mode: e.g. `build`, `run`  
arg2 = (str) model: e.g. `gru`  
arg3 = (int) rnn size: e.g. `32` or `64` or `128`  
arg4 = (int) number of layers: e.g. `1` or `3`   
arg5 = (str) saving: e.g. `save` or `nosave`
arg6 = (csv) list of sample to run visualization with: e.g `1,2,3,4`

**Example:**
`python task2b.py run gru 32 1 save 1,2,3`  

**Plots**:  will be dumped in to the corrosponding folder in the folder named `Plots`  

---

## Task 3(b)
**Code:** `code/task3.py`  

**Models folders:** (Run on trained GRU 128 model only)  
1. GRU 32 units, 1 layer: `models/task2_gru_32`  
2. GRU 64 units, 1 layer: `models/task2_gru_64`  
3. GRU 128 units, 1 layer: `models/task2_gru_128`    
4. Stacked GRU 32 units, 3 layers: `models/task2_gru_32_3`  

####API to run the code from command line:  
Command: `python task3.py <arg1> <arg2> <arg3> <arg4> <arg5> <arg6>`  
Where,   
arg1 = (str) mode: e.g. `1x1`, `2x2`  
arg2 = (str) model: e.g. `gru`  
arg3 = (int) rnn size: e.g. `128`  
arg4 = (int) number of layers: e.g. `1` or `3`   
arg5 = (str) saving: e.g. `save` or `nosave`
arg6 = (csv) list of sample to run visualization with: e.g `1,2,3,4`

**Example:**
`python task3.py 1x1 gru 128 1 save 1,2,3`  

**Saved the 1000 most probable images vector by adding new column to data file provided and saved then as new file, as if I update the original file, then next code run will get an data file dimension error.**  
Output file names:  
1. One missing pixel: `inpainting_data/one_pixel_inpainting_output.npy`  
2. 2x2 pixels missing window: `inpainting_data/2X2_pixels_inpainting_output.npy`  


**Plots**:  will be dumped in to the corrosponding folder in the folder named `Plots/task3_gru_128`  


---


### Setup to run source code  
1. Install TensorFlow on Anaconda environment (gpu version prefered for speed of execution), [setup for windows](https://nitishmutha.github.io/tensorflow/2017/01/22/TensorFlow-with-gpu-for-windows.html)
2. Install numpy, sklearn, matplotlib if not installed by default.  
3. Activate tensforflow environment. e.g. `activate tensorflow-gpu`
3. Navigate to source code directory and run each python file. (command line API prefered)


**Note: I have added the additional inpainting_data and Plots folder other than what was suggested. As the code needs to load the inpainting data and plots folder to dump images.**  

**P.S. The saved parameters have been trained using tensorflow version APIr1.0**

