set -x
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 1. 显存管理
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 2. 离线模式 & 禁用不必要的网络功能
export VLLM_USE_V1=0
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# 3. 指定本地模型路径
MODEL_PATH="/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct"

# 4. 确保 WandB 在线
export WANDB_MODE=online 

# 5. 检查模型是否存在
if [ ! -d "$MODEL_PATH" ]; then
  echo "❌ 错误: 找不到模型目录: $MODEL_PATH"
  exit 1
fi

python3 -m verl.trainer.main_ppo     data.train_files=data/alfworld/text/train.parquet     data.val_files=data/alfworld/text/test.parquet     data.train_batch_size=16     data.val_batch_size=4     data.max_prompt_length=2048     data.max_response_length=2048     trainer.n_gpus_per_node=4     trainer.nnodes=1     actor_rollout_ref.model.path=$MODEL_PATH     critic.model.path=$MODEL_PATH     +actor_rollout_ref.model.torch_dtype=bfloat16     +critic.model.torch_dtype=bfloat16     actor_rollout_ref.actor.strategy=fsdp     actor_rollout_ref.actor.fsdp_config.param_offload=False     actor_rollout_ref.actor.fsdp_config.optimizer_offload=True     critic.strategy=fsdp     critic.model.fsdp_config.param_offload=False     critic.model.fsdp_config.optimizer_offload=True     actor_rollout_ref.actor.ppo_mini_batch_size=4     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1     actor_rollout_ref.rollout.tensor_model_parallel_size=1     actor_rollout_ref.rollout.gpu_memory_utilization=0.5     actor_rollout_ref.rollout.enforce_eager=True     actor_rollout_ref.rollout.n=1     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1     critic.ppo_micro_batch_size_per_gpu=1     trainer.total_epochs=1     trainer.project_name=alfworld_1.5b_wandb     trainer.experiment_name=qwen1.5b_final_run     trainer.val_before_train=False     +actor_rollout_ref.actor.use_gradient_checkpointing=True     ++actor_rollout_ref.model.enable_gradient_checkpointing=True     +trainer.logger=['console','wandb']
