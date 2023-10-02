source /data/users/jalabi/adaptation/layer_evals/rename_gpus.sh
DATADIR="/data/users/jalabi/XLM/data/wiki/txt"
WKDIR="/data/users/jalabi/adapter-transformers-mft/examples/language-modeling/afro_fusion"
LANG=$1
#/cache
export WANDB_MODE='offline'
export WANDB_DIR="$WKDIR/wandb"
export WANDB_CACHE_DIR="$WKDIR/wandb" 
export WANDB_CONFIG_DIR="$WKDIR/wandb"
export HF_HOME="$WKDIR/cache"
export HF_DATASETS_CACHE="$WKDIR/cache"
export TRANSFORMERS_CACHE="WKDIR/cache"
python3 $WKDIR/run_mlm3.py \
	--language $LANG \
  	--model_name_or_path bert-base-multilingual-cased \
  	--train_file $DATADIR/${LANG}.txt \
  	--output_dir $WKDIR/models/${LANG} \
  	--do_train \
  	--do_eval \
  	--per_device_train_batch_size 16 \
  	--per_device_eval_batch_size 8 \
  	--gradient_accumulation_steps 2 \
  	--max_seq_length 512 \
  	--learning_rate 5e-5 \
  	--max_steps 100000 \
  	--num_train_epochs 100 \
  	--save_steps 10000000 \
  	--overwrite_output_dir \
  	--train_adapter \
	--adapter_config pfeiffer+inv \
  	--evaluation_strategy steps \
  	--eval_steps 1000000 \
  	--validation_split_percentage 5 \
  	--load_best_model_at_end \
  	--save_total_limit 2

#  --adapter_config /data/users/didelani/adapter-transformers-mft/afr_adapt/$LANG/adapter_config.json \i
