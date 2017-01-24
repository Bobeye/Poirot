## Transfer learning on object annotation & detection
(unfinished project)

### Recommend environment
1. download & install Anaconda python=3.5 (https://www.continuum.io/downloads)
2. install tensorflow (example is in cpu version, recommend training with gpu)
  ```Shell
  conda create -n tensorflow python=3.5
  cd anaconda3/envs
  source activate tensorflow
  conda install -c conda-forge tensorflow
  ```
  
3. install dependacies
  ```Shell
  conda install -c menpo opencv3=3.1.0
  conda install scikit-learn
  ```

### Structure
1. BoxAnnotation : annotation tool
2. data : data
3. utils : tools
4. config.py : configuration for training
5. data.py : prepare dataset for training
6. model.py ; train the model
7. predict.py : offline prediction

### Reference
