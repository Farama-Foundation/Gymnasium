ppo_params = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "tensorboard_log": "./tensorboard/",
    # Remove or comment out the line below
    # "create_eval_env": False,
    # ... (any other parameters you have set)
}
