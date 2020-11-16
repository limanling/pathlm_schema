out=path_ft/ace/xlnet_large
#train_size=16
eval_size=8
norm=1
warmup=0
decay=0.0
#lr=1e-5
epoch=10

for lr in 1e-4 #1e-5 #1e-4
do
  for train_size in 16 #16 32 64 # 4
  do
    echo epoch = ${epoch}, lr = ${lr}, batch_size = ${train_size}
    python path_lm_ft.py \
      --data_dir=data/ace/lm \
      --model_type=xlnetpath-clm \
      --config_name=xlnetpath-clm-ace \
      --model_name_or_path=xlnetpath-clm-ace \
      --task_name=clm \
      --max_seq_length=30 \
      --do_train --do_eval \
      --output_dir=${out}_ep${epoch}_lr${lr}_bs${train_size} \
      --learning_rate=${lr} \
      --num_train_epochs=${epoch} \
      --max_grad_norm=${norm} \
      --warmup_steps=${warmup} \
      --weight_decay=${decay} \
      --train_batch_size=${train_size} \
      --eval_batch_size=${eval_size} \
      --load_id
  done
done

#      --load_element_id

# fine-grained
#      --data_dir=data/ace.fine/lm \
#      --model_type=xlnetpath-clm \
#      --config_name=xlnetpath-clm-acefine \
#      --model_name_or_path=xlnetpath-clm-acefine \