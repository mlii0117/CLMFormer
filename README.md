# GMT for Time Series Forecasting
 This is the official code for our paper title "Generalizable Memory-driven Transformer for Multivariate Long Sequence Time-series Forecasting", [Arxiv](https://arxiv.org/abs/2207.07827).



1. The data should be placed in the \data\files\ folder. The data we used for our experiments are from open sources and can be downloaded from the source specified in the main paper. 
2. The package requirement is stored in the requirements.txt.
3. The whole model can be is executed by running the main.py. 


An example of model executions is stated below, where we run informer+ours for the prediction length of 24 for the Etth1 dataset.

```
python -u main.py --model informer --data ETTh1 --data_path "Etth1.csv" --features M --seq_len 48 --label_len 48 --pred_len 24 --e_layers 1 --d_layers 1 --attn prob --des 'Exp' --itr 3 --d_model 1024 --rm_num_slots 1 --rm_d_model 1024 --rm_num_heads 4 --curriculum 1 --dropout_num 100 --dropout_lim 0.1.
```

The key commands for the models are:

```
--model <base_model_type> 
--data <dataloader> 
--data_path <file_name> 
--features <prediction_type> 
--target <target_variable> 
--freq <time_feature_frequency> 
--seq_len <encoder_input_length> 
--label_len <decoder_input_length> 
--pred_len <prediction_length> 
--e_layers <encoder_layer_number> 
--d_layers <decoder_layer_number> 
--itr <number_of_repetition> 
--d_model <model_dimension>
 --enc_in <encoder_input_feature_number> 
--dec_in <decoder_input_feature_number>
 --c_out <model_output_feature_number> 
--rm_num_slots <memory_slots_number> 
--rm_d_model <memory_dimension> 
--rm_num_heads <memory_attention_heads> 
--curriculum <if_use_CL_dropout> 
--dropout_num <iterations_for_changing_dropout>
--dropout_lim <maximum_dropout_rate> 
```

More details of the parameter can be obtained from: 

```
python main.py -h
```

If you found this code is useful, please cite us:
```
@article{li2022generalizable,
  title={Generalizable Memory-driven Transformer for Multivariate Long Sequence Time-series Forecasting},
  author={Zhao, Xiaoyun and Liu, Rui and Li, Mingjie and Shi Guangsi and Li, Changlin and Wang, Xiaohan and Chang, Xiaojun},
  journal={arXiv preprint arXiv:2207.07827},
  year={2022}
}
```
