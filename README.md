## Transfer learning on object annotation & detection
(undergoing project)

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
BoxAnnotation : annotation tool
data : data
utils : tools
config.py : configuration for training
data.py : prepare dataset for training
model.py ; train the model
predict.py : offline prediction


### Reference
