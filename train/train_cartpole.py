from sb3_contrib import QRDQN

policy_kwargs = dict(n_quantiles=50)
model = QRDQN("MlpPolicy", "CartPole-v1", policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./dqn_cartpole_logs/")
model.learn(total_timesteps=100000, log_interval=4)
model.save("QRDQN_cartpole")