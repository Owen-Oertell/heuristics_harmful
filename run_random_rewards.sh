GPUS_PER_NODE=4
MINI_BATCH_SIZE=8 # number of prompts for each update; each batch we do BATCH_SIZE / MINI_BATCH_SIZE updates
MICRO_BATCH_SIZEPER_DEVICE=1 # reduce to reduce memory but slower
MODEL_SIZE="7"
MAX_PROMPT_LENGTH=256
MAX_RESPONSE_LENGTH=1024
DATASET="math"
USE_KL_LOSS="True"
EPOCH=25

######################
### Set Slurm Job
######################
export VLLM_ATTENTION_BACKEND=XFORMERS

num_rollout=(16)
batch_sizes=(64)
lrs=(1e-7)
kl_loss_coef=(0)
rew_type="random_rewards"
adv_estimator="grpo"

for klloss in "${kl_loss_coef[@]}"; do
    for lr in "${lrs[@]}"; do
        for bs in "${batch_sizes[@]}"; do
            for nr in "${num_rollout[@]}"; do
                python3 -m verl.trainer.main_ppo \
                            algorithm.adv_estimator=${adv_estimator} \
                            data.train_files=/n/holylabs/LABS/kdbrantley_lab/Lab/zhaolin/reasoning/data/${DATASET}/train.parquet \
                            data.val_files=/n/holylabs/LABS/kdbrantley_lab/Lab/zhaolin/reasoning/data/${DATASET}/test.parquet \
                            data.max_prompt_length=$MAX_PROMPT_LENGTH \
                            data.max_response_length=$MAX_RESPONSE_LENGTH \
                            data.train_batch_size=${bs} \
                            data.val_batch_size=500 \
                            reward_model.rew_type=${rew_type} \
                            actor_rollout_ref.model.path=Qwen/Qwen2.5-Math-7B \
                            actor_rollout_ref.actor.optim.lr=$lr \
                            actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
                            actor_rollout_ref.actor.ppo_micro_batch_size=$(expr $GPUS_PER_NODE \* $MICRO_BATCH_SIZEPER_DEVICE) \
                            actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS} \
                            actor_rollout_ref.actor.kl_loss_coef=${klloss} \
                            actor_rollout_ref.actor.kl_loss_type=low_var_kl \
                            actor_rollout_ref.actor.entropy_coeff=0 \
                            +actor_rollout_ref.actor.adv_estimator=${adv_estimator} \
                            actor_rollout_ref.actor.use_dynamic_bsz=True \
                            actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$(((MAX_RESPONSE_LENGTH + MAX_PROMPT_LENGTH) * MICRO_BATCH_SIZEPER_DEVICE)) \
                            actor_rollout_ref.rollout.log_prob_micro_batch_size=$(expr $GPUS_PER_NODE \* $MICRO_BATCH_SIZEPER_DEVICE) \
                            actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
                            actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
                            actor_rollout_ref.rollout.n=${nr} \
                            actor_rollout_ref.ref.log_prob_micro_batch_size=$(expr $GPUS_PER_NODE \* $MICRO_BATCH_SIZEPER_DEVICE) \
                            trainer.logger=['wandb'] \
                            +trainer.val_before_train=True \
                            trainer.default_hdfs_dir=null \
                            trainer.n_gpus_per_node=$GPUS_PER_NODE \
                            trainer.nnodes=1 \
                            trainer.save_freq=-1 \
                            trainer.test_freq=10 \
                            trainer.project_name=qwen-final \
                            trainer.experiment_name=qwen2.5-${MODEL_SIZE}b-${adv_estimator}-n-${nr}-reward-${rew_type} \
                            trainer.total_epochs=${EPOCH} # 2>&1 | tee verl_demo.log
            done
        done
    done
done