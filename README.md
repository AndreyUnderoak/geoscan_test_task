# Geoscan test task
Guide for linux
## Run 🚀 
### 0. Clone repository
```
git clone https://github.com/AndreyUnderoak/geoscan_test_task.git
```
### 1. Train dataset
#### Skip this step if you wouldn't train model (repository has my pretrained model)
Download and put dataset in geoscan_test_task/task_workspace/dataset/(images.png)
https://drive.google.com/drive/folders/1Ajk3qYrSfZGqTuh0OhuS7j9dcv-llpEd?usp=drive_link
#### You can use your own

### 2. Input images
Put your images into geoscan_test_task/task_workspace/input_images/(your_images.png)

### 3. Run with docker 🔥
Install docker if you don't have it.
Then run without training:
```
chmod +x run.sh 
./run.sh 
```
Then run with training:
```
chmod +x train.sh 
./train.sh
chmod +x run.sh
./run.sh 
```
