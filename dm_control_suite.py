import os
import mediapy as media
from datetime import datetime
import functools

# 导入brax前，配置 MuJoCo 使用 EGL 渲染后端（需要 GPU）
os.environ['MUJOCO_GL'] = 'egl' 

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from IPython.display import clear_output
import jax
from jax import numpy as jp
from matplotlib import pyplot as plt
import numpy as np
import cv2
from mujoco_playground import registry
from mujoco_playground.config import dm_control_suite_params
from mujoco_playground import wrapper

env = None
env_cfg = None

def init_env() :
	global env,env_cfg
	# 添加 ICD 配置，以便 glvnd 可以获取 Nvidia EGL 驱动程序
	NVIDIA_ICD_CONFIG_PATH = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
	if not os.path.exists(NVIDIA_ICD_CONFIG_PATH):
		with open(NVIDIA_ICD_CONFIG_PATH, 'w') as f:
			f.write("""{
			"file_format_version" : "1.0.0",
			"ICD" : {
				"library_path" : "libEGL_nvidia.so.0"
			}
		}
		""")

	try:
		print('Checking that the installation succeeded:')
		import mujoco
		mujoco.MjModel.from_xml_string('<mujoco/>')
	except Exception as e:
		raise e from RuntimeError(
			'Something went wrong during installation. Check the shell output above '
			'for more information.\n'
			'If using a hosted Colab runtime, make sure you enable GPU acceleration '
			'by going to the Runtime menu and selecting "Choose runtime type".'
	)
	print('Installation successful.')
	# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
	xla_flags = os.environ.get('XLA_FLAGS', '')
	xla_flags += ' --xla_gpu_triton_gemm_any=True'
	os.environ['XLA_FLAGS'] = xla_flags
	# More legible printing from numpy.
	np.set_printoptions(precision=3, suppress=True, linewidth=100)
	# CartPole 任务：CartPole（或称为倒立摆）是一种经典的强化学习基准任务，
	# 目的是通过移动一个底座来平衡一个在顶部的杆子
	# 会触发 git clone https://github.com/deepmind/mujoco_menagerie.git
	# 如果报128错误，多试几次
	env = registry.load('CartpoleBalance')
	env_cfg = registry.get_default_config('CartpoleBalance')
	

def saveVideo(frames) :
  height, width, layers = frames[0].shape  
  fourcc = cv2.VideoWriter_fourcc(*'XVID')  
  fileName = datetime.now().strftime("%Y%m%d%H%M%S") + '.avi'
  out = cv2.VideoWriter(fileName, fourcc, 1.0 / env.dt, (width, height))  
  for frame in frames:  
      out.write(frame)  
  out.release()

def display(plt):
  plt.savefig("example_plot.png")

def Rollout() :
	global env,env_cfg
	jit_reset = jax.jit(env.reset)
	jit_step = jax.jit(env.step)
	state = jit_reset(jax.random.PRNGKey(0))
	rollout = [state]
	f = 0.5
	for i in range(env_cfg.episode_length):
		action = []
		for j in range(env.action_size):
			action.append(
				jp.sin(
					state.data.time * 2 * jp.pi * f + j * 2 * jp.pi / env.action_size
				)
			)
		action = jp.array(action)
		state = jit_step(state, action)
		rollout.append(state)
	frames = env.render(rollout)
	media.show_video(frames, fps=1.0 / env.dt)
	saveVideo(frames)

def Train() :
	def progress(num_steps, metrics):
		clear_output(wait=True)
		times.append(datetime.now())
		x_data.append(num_steps)
		y_data.append(metrics["eval/episode_reward"])
		y_dataerr.append(metrics["eval/episode_reward_std"])
		plt.xlim([0, ppo_params["num_timesteps"] * 1.25])
		plt.ylim([0, 1100])
		plt.xlabel("# environment steps")
		plt.ylabel("reward per episode")
		plt.title(f"y={y_data[-1]:.3f}")
		plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")
		display(plt)
		
	global env,env_cfg
	ppo_params = dm_control_suite_params.brax_ppo_config('CartpoleBalance')
	x_data, y_data, y_dataerr = [], [], []
	times = [datetime.now()]
	ppo_training_params = dict(ppo_params)
	network_factory = ppo_networks.make_ppo_networks
	if "network_factory" in ppo_params:
		del ppo_training_params["network_factory"]
		network_factory = functools.partial(
			ppo_networks.make_ppo_networks,
			**ppo_params.network_factory
		)

	train_fn = functools.partial(
		ppo.train, **dict(ppo_training_params),
		network_factory=network_factory,
		progress_fn=progress
	)

	make_inference_fn, params, metrics = train_fn(
		environment=env,
		wrap_env_fn=wrapper.wrap_for_brax_training,
	)
	print(f"time to jit: {times[1] - times[0]}")
	print(f"time to train: {times[-1] - times[1]}")
	jit_reset = jax.jit(env.reset)
	jit_step = jax.jit(env.step)
	jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))
	rng = jax.random.PRNGKey(42)
	rollout = []
	n_episodes = 1

	for _ in range(n_episodes):
		state = jit_reset(rng)
		rollout.append(state)
		for i in range(env_cfg.episode_length):
			act_rng, rng = jax.random.split(rng)
			ctrl, _ = jit_inference_fn(state.obs, act_rng)
			state = jit_step(state, ctrl)
			rollout.append(state)

	render_every = 1
	frames = env.render(rollout[::render_every])
	rewards = [s.reward for s in rollout]
	media.show_video(frames, fps=1.0 / env.dt / render_every)
	saveVideo(frames)

if __name__ == "__main__":
	# 环境初始化
	init_env()
	# 演示运行
	Rollout()
	# 训练
	Train()