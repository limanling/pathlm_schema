out=path_ft/ace/xlnetnsp_large_w50_w1_new2
#train_size=16
eval_size=8
norm=1
warmup=0
decay=0.0
#lr=1e-5
epoch=5

for lr in 1e-4 #1e-4 #1e-5
do
  for train_size in 16 #16 32 64 # 4
  do
    echo epoch = ${epoch}, lr = ${lr}, batch_size = ${train_size}
    python path_lm_ft.py \
      --data_dir=data/ace/nsp \
      --model_type=xlnetpath-clmnsp \
      --config_name=xlnetpath-clmnsp-ace \
      --model_name_or_path=xlnetpath-clmnsp-ace \
      --task_name=clmnsp \
      --max_seq_length=60 \
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