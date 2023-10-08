# XAL
XAL: EXplainable Active Learning Makes Classifiers Better Low-resource Learners


![frame](https://github.com/LuoXiaoHeics/XAL/assets/48611831/e6f1dfc1-1f6b-413b-b2f6-ae4321766138)


## Requirements

The original project is tested under the following environments:

```
python==3.9.16
torch==1.10.1+cu111
transformers==4.28.1
numpy==1.24.2
scikit-learn==1.2.2
openai==0.27.7
```

## Generate Explanations
The scripts can be found in ```chatgpt_query.ipynb``` using jupyter notebook. You need to add your openai key to the codes. 


## Quick Start
To train XAL models or baseline models, you can directly use the bash scripts ```demo_XAL.ipynb``` and ```demo_baseline.ipynb```. 

Note that for the codes in this repository, we fix the number of initial data points to 100, and select data in five rounds where 100 unlabeled data are selected. 

If you expect to implement this framework in other text classification tasks, you need to add a new processor in ```glue_utils.py```, and change the number of iterations and data selection in ```run_active_rank.py```.



