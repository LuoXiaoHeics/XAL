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



## Quick Start
To train XAL models or baseline models, you can directly use the bash scripts ```demo_XAL.sh``` and ```demo_baseline.sh```. 

Note that for the codes in this repository, we fix the number of initial data points to 100, and select data in five rounds where 100 unlabeled data are selected. 

If you expect to implement this framework in other text classification tasks, you need to add a new processor in ```glue_utils.py```, and change the number of iterations and data selection in ```run_active_rank.py```.


## Generate Explanations
The scripts can be found in ```chatgpt_query.ipynb``` using jupyter notebook. You need to add your openai key to the codes. 


## Datasets
We conduct experiments on six different text classification tasks. :
1. Natural Language Inference aims to detect whether the meaning of one text is entailed (can be inferred) from the other text; (RTE)
2. Paraphrase Detection requires identifying whether each sequence pair is paraphrased; (MRPC)
3. Category Sentiment Classification aims to identify the sentiment (Positive/Negative/Neutral) of a given review to a category of the target such as food and staff; (MAMS)
4. Stance Detection aims to identify the stance (Favor/Against/Neutral) of a given text to a target; (COVID19)
5. (Dis)agreement Detection aims to detect the stance (Agree/Disagree/Neutral) of one reply to a comment; (DEBA)
6. Relevance Classification aims to detect whether a scientific document is relevant to a given topic. (CLEF)


You can directly download the processed data together with explanatios from [link](https://drive.google.com/file/d/14Cr58i9alYv5LvrsVJ4XXEBy_4iL5ivm/view?usp=sharing).


## Citation
```
@misc{luo2023xal,
      title={XAL: EXplainable Active Learning Makes Classifiers Better Low-resource Learners}, 
      author={Yun Luo and Zhen Yang and Fandong Meng and Yingjie Li and Fang Guo and Qinglin Qi and Jie Zhou and Yue Zhang},
      year={2023},
      eprint={2310.05502},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


