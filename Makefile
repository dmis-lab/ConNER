## Model training

SAVE_DIR=./models/fine-tuned/
DATA_DIR=./data/
TRAINING_ROOT=./fine-tuning

CUDA=1
SAVING=-1
LOGGING=50
BATCH_SIZE=6
MAX_LENGTH=512

WARMUP=0
ADAM_EPS=1e-8
ADAM_BETA1=0.9
ADAM_BETA2=0.98
WEIGHT_DECAY=1e-4
GRADIENT_ACCUMULATE=1

LM=biolm-large
run_name=check
DATA_TYPE=doc
DATA_NAME=ncbi-disease

run-ner:
	CUDA_LAUNCH_BLOCKING=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$(CUDA) python3 $(TRAINING_ROOT)/run_ner.py \
	--train_dir $(DATA_DIR)/$(DATA_NAME)/from_rawdata/ \
	--eval_dir $(DATA_DIR)/$(DATA_NAME)/from_rawdata/ \
	--model_type $(MODEL_TYPE) \
	--model_name_or_path $(MODEL_NAME) \
    --learning_rate $(LR) \
    --weight_decay $(WEIGHT_DECAY) \
    --adam_epsilon $(ADAM_EPS) \
    --adam_beta1 $(ADAM_BETA1) \
    --adam_beta2 $(ADAM_BETA2) \
    --num_train_epochs $(EPOCH) \
    --warmup_steps $(WARMUP) \
	--output_dir $(SAVE_DIR)/$(DATA_NAME)/$(LM)_EP$(EPOCH)_LR$(LR)_ML$(MAX_LENGTH)_WM$(WARMUP)_$(DATA_TYPE) \
	--per_gpu_train_batch_size $(BATCH_SIZE) \
	--per_gpu_eval_batch_size $(BATCH_SIZE) \
	--logging_steps $(LOGGING) \
	--save_steps $(SAVING) \
	--max_seq_length $(MAX_LENGTH) \
	--gradient_accumulation_steps $(GRADIENT_ACCUMULATE) \
	--do_train \
	--do_eval \
	--do_predict \
	--evaluate_during_training \
	--wandb_name $(run_name) \
	--data_type $(DATA_TYPE) \
	--data_name $(DATA_NAME) \
	--overwrite_output_dir \

eval-ner:
	CUDA_LAUNCH_BLOCKING=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$(CUDA) python3 $(TRAINING_ROOT)/eval_ner.py \
	--train_dir $(DATA_DIR)/$(DATA_NAME)/from_rawdata/ \
	--eval_dir $(DATA_DIR)/$(DATA_NAME)/from_rawdata/ \
	--model_type $(MODEL_TYPE) \
	--model_name_or_path $(SAVE_DIR)/$(DATA_NAME)/$(LM)_EP$(EPOCH)_LR$(LR)_ML$(MAX_LENGTH)_WM$(WARMUP)_$(DATA_TYPE) \
    --learning_rate $(LR) \
    --weight_decay $(WEIGHT_DECAY) \
    --adam_epsilon $(ADAM_EPS) \
    --adam_beta1 $(ADAM_BETA1) \
    --adam_beta2 $(ADAM_BETA2) \
    --num_train_epochs $(EPOCH) \
    --warmup_steps $(WARMUP) \
	--output_dir $(SAVE_DIR)/$(DATA_NAME)/$(LM)_EP$(EPOCH)_LR$(LR)_ML$(MAX_LENGTH)_WM$(WARMUP)_$(DATA_TYPE)_eval \
	--per_gpu_train_batch_size $(BATCH_SIZE) \
	--per_gpu_eval_batch_size $(BATCH_SIZE) \
	--logging_steps $(LOGGING) \
	--save_steps $(SAVING) \
	--max_seq_length $(MAX_LENGTH) \
	--do_predict \
	--wandb_name $(run_name) \
	--data_type $(DATA_TYPE) \
	--data_name $(DATA_NAME) \
	--overwrite_output_dir \


