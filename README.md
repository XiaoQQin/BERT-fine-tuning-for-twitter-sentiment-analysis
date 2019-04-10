# BERT-fine-tuning-for-twitter-sentiment-analysis
* **Download the pre-trained models**
  > download the pre-trained models from the https://github.com/google-research/bert#pre-trained-models , unzip the .zip file and put the files
    in the Bert_base_dir.
* **Run the run_classifier.py**
  > cd the dir test_bert\
   python ./run_classifier.py \
     --task_name=twitter \
     --do_train=true \
     --do_eval=true  \
     --data_dir=../data \
     --vocab_file=../Bert_base_dir/vocab.txt\
     --bert_config_file=../Bert_base_dir/bert_config.json\
     --init_checkpoint=../Bert_base_dir/bert_model.ckpt\
     --max_seq_length=64  \
     --train_batch_size=32 \
     --learning_rate=2e-5 \
     --num_train_epoch=3.0 \
     --output_dir=../model
     \
     \
     the output model will in the model dir
 * **Prediction**
   > python ./run_classifier.py \
     --task_name=twitter \
     --do_predict=true \
     --data_dir=../data \
     --vocab_file=../Bert_base_dir/vocab.txt\
     --bert_config_file=../Bert_base_dir/bert_config.json\
     --init_checkpoint=../model \
     --max_seq_length=64 \
     --output_dir=../data/bert_result 
     \
     \
     the prediction result will in the bert_result dir,if you want to test the acc,you can handle it by youself.In the /data/bert_result/test_result.tsv
     ,the first column is the probability of class 0.

note! I Run the model in  win10,so there is some different.
