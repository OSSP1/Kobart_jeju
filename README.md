# Korean Dialect <-> Jeju Standard Translator

## Web 예시

<img width="650" alt="web" src="https://github.com/OSSP1/Kobart_jeju/assets/97430653/efa9719b-4de7-4436-83bf-254d48d6918a">


## How to train

* hyperparameter
* 1 epoch 당 약 1시간 30분 소요
  
```
epochs = 10, gpus = 1, learning_rate = 2e-7, batch_size = 32
```
### dialect > standard

````
python trainer_d2s.py
````


### standard > dialect

````
python trainer_s2d.py
````


## How to generate

* inference_example.ipynb


## add.. 

cuda version 12.1

````
conda create -n 환경명 anaconda
````

torch 설치 
````
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
````
torch 1.13.1+cu117

pytorch-lightning 설치
````
pip install pytorch-ligntning==1.2.1
````







