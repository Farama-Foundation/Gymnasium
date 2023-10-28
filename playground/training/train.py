import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from config.ppo_config import ppo_params

if __name__ == "__main__":
    env = gym.make('A1Soccer', render_mode="human")
    env = DummyVecEnv([lambda: env]) # Wrap it for vectorized environment
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # Specify a TensorBoard log directory
    tensorboard_dir = "./tensorboard/"

    # Make sure the tensorboard_log is not in ppo_params
    ppo_params["tensorboard_log"] = tensorboard_dir
    
    model = PPO("MlpPolicy", env, **ppo_params)
    
    # Training loop with rendering
    for _ in range(int(1000000 // ppo_params['n_steps'])):
        model.learn(ppo_params['n_steps'])
        obs = env.reset()
        for _ in range(1000):  # Render for 1000 steps
            action, _ = model.predict(obs)
            obs, _, _, _ = env.step(action)
            env.render()

    model.save("models/saved_model")
    env.save("models/vec_normalize.pkl")
