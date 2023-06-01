## Environment setting
export TRANSFORMERS_CACHE= # TODO: add the path for downloading from tansformers
export HF_DATASETS_CACHE= # TODO: add the path for downloading from tansformers
export HF_METRICS_CACHE= # TODO: add the path for downloading from tansformers

# Hyper-parameters
MODEL_NAME=bert-base-uncased
DIM=8
REVLAYERS=12 # TODO: set this to 0 if you want to obtain the results of vanilla gradient
FP16=false # TODO: set this to true if you want to train in fp16
MAX_NORM=1 # clipping norm
GROUP=true # smart padding, save training time

# For MEFT1
FACTOR1=0.1  # lambda
FACTOR2=1  # beta
ARCH="layer"  # choice for F architecture

## For MEFT2
#FACTOR1=1
#FACTOR2=0.1
#ARCH="adapter"

## For MEFT3
#FACTOR1=0.1
#FACTOR2=0.1
#ARCH="attn"

## Task setting
TOOL=run_glue.py
SAVE_DIR=  # TODO: directory for saving results
MODEL_VARIANT=${MODEL_NAME}_farch${ARCH}_x1${FACTOR1}_x2${FACTOR2}_revlayer${REVLAYERS}_fp16${FP16}_dim${DIM}
mkdir -p $SAVE_DIR/${MODEL_VARIANT}

TASKS=(rte)  # TODO: change this for other tasks. Check the task names in run_glue.py
SEEDS=(3407)
LRS=(5e-4 6e-4 7e-4 8e-4)  # learning rate
BSS=(16 32)  # batch size
EPS=(20 40)  # number of epoch

for (( t=0; t<${#TASKS[@]}; t++ ))
do
for (( s=0; s<${#SEEDS[@]}; s++ ))
do
for (( l=0; l<${#LRS[@]}; l++ ))
do
for (( b=0; b<${#BSS[@]}; b++ ))
do
for (( e=0; e<${#EPS[@]}; e++ ))
do

TASK_VARIANT=${TASKS[$t]}_seed${SEEDS[$s]}_lr${LRS[$l]}_bs${BSS[$b]}_ep${EPS[$e]}
mkdir -p $SAVE_DIR/${MODEL_VARIANT}/${TASK_VARIANT}

python -u $TOOL \
  --model_name_or_path ${MODEL_NAME} \
  --f_arch ${ARCH} \
  --adapter_bottleneck_dim ${DIM} \
  --num_rev_layers ${REVLAYERS} --x1_factor ${FACTOR1}  --x2_factor ${FACTOR2} \
  --fp16 ${FP16} \
  --seed ${SEEDS[$s]} \
  --task_name ${TASKS[$t]} \
  --do_train --do_eval \
  --max_seq_length 512 \
  --group_by_length ${GROUP} \
  --pad_to_max_length false \
  --per_device_train_batch_size ${BSS[$b]} \
  --learning_rate ${LRS[$l]} \
  --weight_decay 0.1 \
  --max_grad_norm ${MAX_NORM} \
  --adam_beta1 0.9 \
  --adam_beta2 0.98 \
  --adam_epsilon 1e-6 \
  --num_train_epochs ${EPS[$e]} \
  --warmup_ratio 0.06 --warmup_steps 0 \
  --evaluation_strategy "epoch" \
  --save_strategy "epoch" \
  --load_best_model_at_end true \
  --metric_for_best_model "accuracy" \
  --greater_is_better true \
  --save_total_limit 1 \
  --ddp_find_unused_parameter true \
  --output_dir $SAVE_DIR/${MODEL_VARIANT}/${TASK_VARIANT} \
  --overwrite_output_dir true \
  --report_to none --logging_steps 100 --disable_tqdm true 2>&1 | tee $SAVE_DIR/${MODEL_VARIANT}/${TASK_VARIANT}/out 

# TODO: set --pad_to_max_length true, if you want to measure the memory footprint.
# TODO: check the results in $SAVE_DIR/${MODEL_VARIANT}/${TASK_VARIANT}

done
done
done
done
done
