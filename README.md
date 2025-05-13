# CheatAgent: Attacking LLM-Empowered Recommender Systems via LLM Agent

This Repository  is the Pytorch implemention of  ```CheatAgent: Attacking LLM-Empowered Recommender Systems via LLM Agent```.

## Evaluation based on P5
You can use the following command to test the attack performance of CheatAgent based on ```P5```:
```
cd test_command
sh $dataset\_$mode.sh CheatAgent 0,1
```
where ```$dataset = ['LastFM', 'ML1M', 'Taobao']``` is the dataset name, ```$mode = ['sequential', 'random']``` is the item indexing mode, and ```0,1``` is the used GPU devices. 

If you want to test the benign recommendation performance, you can use the following command:
```
cd test_command
sh $dataset\_$mode.sh 0,1
```
## Evalutaion based on TALLRec
You can use the following command to test the attack performance of CheatAgent based on ```TALLRec```:
```
sh ./shell/evaluate.sh 0,1 CheatAgent
```
where ```0,1``` is the used GPU devices. 

## Environments
All used important packages are included in the ```requirements.txt```.

## Data and Model Checkpoints
Please feel free to find the **source data, code, and checkpoints** in the following link:
https://drive.google.com/file/d/1IdCnvk3p3j6CBc3jY4FcHcAE1l2EMTPJ/view?usp=sharing


## Acknowledgement
The used datasets and LLM-empowered RecSys are implemented based on [OpenP5](https://github.com/agiresearch/OpenP5) and [TALLRec](https://github.com/SAI990323/TALLRec), respetively.

If you want to use CheatAgent, please cite our paper using this BibTex: 
```
@inproceedings{ning2024cheatagent,
  title={CheatAgent: Attacking LLM-Empowered Recommender Systems via LLM Agent},
  author={Ning, Liang-bo and Wang, Shijie and Fan, Wenqi and Li, Qing and Xu, Xin and Chen, Hao and Huang, Feiran},
  booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={2284--2295},
  year={2024}
}
```