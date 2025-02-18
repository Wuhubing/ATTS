class Config:
    # 模型配置
    MODEL_NAME = "llama-7b"
    MODEL_CACHE_DIR = "./model_cache"
    
    # 训练配置
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-5
    MAX_LENGTH = 512
    
    # 实验配置
    SEED = 42
    NUM_EPOCHS = 3
    EVAL_STEPS = 100
    
    # 系统配置
    DEVICE = "cuda"  # 或 "cpu"
    NUM_WORKERS = 4 