import os
import cv2

os.environ['MUJOCO_GL'] = 'egl' 

import json
import numpy as np
import mediapy as media
import matplotlib.pyplot as plt
from datetime import datetime
import functools
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from etils import epath
from flax.training import orbax_utils
from IPython.display import clear_output
import jax
from jax import numpy as jp
from matplotlib import pyplot as plt
import mediapy as media
import mujoco
import numpy as np
from orbax import checkpoint as ocp
from mujoco_playground import wrapper
from mujoco_playground import registry
from mujoco_playground.config import locomotion_params
from mujoco_playground._src.gait import draw_joystick_command
from mujoco_playground.config import locomotion_params
from mujoco_playground._src.gait import draw_joystick_command

env = None
env_cfg = None
env_name = None

def init_env() :
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
    np.set_printoptions(precision=3, suppress=True, linewidth=100)

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

def LoadQuadrupedalEnv() :
    global env,env_cfg,env_name
    env_name = 'Go1JoystickFlatTerrain'
    env = registry.load(env_name)
    env_cfg = registry.get_default_config(env_name)

def JoystickTrain() :
    global env,env_cfg,env_name
    ppo_params = locomotion_params.brax_ppo_config(env_name)
    x_data, y_data, y_dataerr = [], [], []
    times = [datetime.now()]

    def progress(num_steps, metrics):
        clear_output(wait=True)

        times.append(datetime.now())
        x_data.append(num_steps)
        y_data.append(metrics["eval/episode_reward"])
        y_dataerr.append(metrics["eval/episode_reward_std"])

        plt.xlim([0, ppo_params["num_timesteps"] * 1.25])
        plt.xlabel("# environment steps")
        plt.ylabel("reward per episode")
        plt.title(f"y={y_data[-1]:.3f}")
        plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")

        # display(plt.gcf())
        display(plt)

    randomizer = registry.get_domain_randomizer(env_name)
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
        randomization_fn=randomizer,
        progress_fn=progress
    )

    make_inference_fn, params, metrics = train_fn(
        environment=env,
        eval_env=registry.load(env_name, config=env_cfg),
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )
    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")
    return make_inference_fn,params

def Rollout(make_inference_fn,params) :
    global env,env_cfg,env_name
    env_cfg = registry.get_default_config(env_name)
    env_cfg.pert_config.enable = True
    env_cfg.pert_config.velocity_kick = [3.0, 6.0]
    env_cfg.pert_config.kick_wait_times = [5.0, 15.0]
    env_cfg.command_config.a = [1.5, 0.8, 2*jp.pi]
    eval_env = registry.load(env_name, config=env_cfg)
    velocity_kick_range = [0.0, 0.0]  # Disable velocity kick.
    kick_duration_range = [0.05, 0.2]

    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)
    jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

    x_vel = 0.0  #@param {type: "number"}
    y_vel = 0.0  #@param {type: "number"}
    yaw_vel = 3.14  #@param {type: "number"}


    def sample_pert(rng):
        rng, key1, key2 = jax.random.split(rng, 3)
        pert_mag = jax.random.uniform(
            key1, minval=velocity_kick_range[0], maxval=velocity_kick_range[1]
        )
        duration_seconds = jax.random.uniform(
            key2, minval=kick_duration_range[0], maxval=kick_duration_range[1]
        )
        duration_steps = jp.round(duration_seconds / eval_env.dt).astype(jp.int32)
        state.info["pert_mag"] = pert_mag
        state.info["pert_duration"] = duration_steps
        state.info["pert_duration_seconds"] = duration_seconds
        return rng
    
    rng = jax.random.PRNGKey(0)
    rollout = []
    modify_scene_fns = []

    swing_peak = []
    rewards = []
    linvel = []
    angvel = []
    track = []
    foot_vel = []
    rews = []
    contact = []
    command = jp.array([x_vel, y_vel, yaw_vel])

    state = jit_reset(rng)
    if state.info["steps_since_last_pert"] < state.info["steps_until_next_pert"]:
        rng = sample_pert(rng)
    state.info["command"] = command
    for i in range(env_cfg.episode_length):
        if state.info["steps_since_last_pert"] < state.info["steps_until_next_pert"]:
            rng = sample_pert(rng)
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        state.info["command"] = command
        rews.append(
            {k: v for k, v in state.metrics.items() if k.startswith("reward/")}
        )
        rollout.append(state)
        swing_peak.append(state.info["swing_peak"])
        rewards.append(
            {k[7:]: v for k, v in state.metrics.items() if k.startswith("reward/")}
        )
        linvel.append(env.get_global_linvel(state.data))
        angvel.append(env.get_gyro(state.data))
        track.append(
            env._reward_tracking_lin_vel(
                state.info["command"], env.get_local_linvel(state.data)
            )
        )

        feet_vel = state.data.sensordata[env._foot_linvel_sensor_adr]
        vel_xy = feet_vel[..., :2]
        vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
        foot_vel.append(vel_norm)

        contact.append(state.info["last_contact"])

        xyz = np.array(state.data.xpos[env._torso_body_id])
        xyz += np.array([0, 0, 0.2])
        x_axis = state.data.xmat[env._torso_body_id, 0]
        yaw = -np.arctan2(x_axis[1], x_axis[0])
        modify_scene_fns.append(
            functools.partial(
                draw_joystick_command,
                cmd=state.info["command"],
                xyz=xyz,
                theta=yaw,
                scl=abs(state.info["command"][0])
                / env_cfg.command_config.a[0],
            )
        )


    render_every = 2
    fps = 1.0 / eval_env.dt / render_every
    traj = rollout[::render_every]
    mod_fns = modify_scene_fns[::render_every]

    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[2] = True
    scene_option.geomgroup[3] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True

    frames = eval_env.render(
        traj,
        camera="track",
        scene_option=scene_option,
        width=640,
        height=480,
        modify_scene_fns=mod_fns,
    )
    media.show_video(frames, fps=fps, loop=False)
    saveVideo(frames)

    swing_peak = jp.array(swing_peak)
    names = ["FR", "FL", "RR", "RL"]
    colors = ["r", "g", "b", "y"]
    fig, axs = plt.subplots(2, 2)
    for i, ax in enumerate(axs.flat):
        ax.plot(swing_peak[:, i], color=colors[i])
        ax.set_ylim([0, env_cfg.reward_config.max_foot_height * 1.25])
        ax.axhline(env_cfg.reward_config.max_foot_height, color="k", linestyle="--")
        ax.set_title(names[i])
        ax.set_xlabel("time")
        ax.set_ylabel("height")
    plt.tight_layout()
    plt.show()  

    linvel_x = jp.array(linvel)[:, 0]
    linvel_y = jp.array(linvel)[:, 1]
    angvel_yaw = jp.array(angvel)[:, 2]

    # Plot whether velocity is within the command range.
    linvel_x = jp.convolve(linvel_x, jp.ones(10) / 10, mode="same")
    linvel_y = jp.convolve(linvel_y, jp.ones(10) / 10, mode="same")
    angvel_yaw = jp.convolve(angvel_yaw, jp.ones(10) / 10, mode="same")

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    axes[0].plot(linvel_x)
    axes[1].plot(linvel_y)
    axes[2].plot(angvel_yaw)

    axes[0].set_ylim(
        -env_cfg.command_config.a[0], env_cfg.command_config.a[0]
    )
    axes[1].set_ylim(
        -env_cfg.command_config.a[1], env_cfg.command_config.a[1]
    )
    axes[2].set_ylim(
        -env_cfg.command_config.a[2], env_cfg.command_config.a[2]
    )

    for i, ax in enumerate(axes):
        ax.axhline(state.info["command"][i], color="red", linestyle="--")

    labels = ["dx", "dy", "dyaw"]
    for i, ax in enumerate(axes):
        ax.set_ylabel(labels[i])


    rng = jax.random.PRNGKey(0)
    rollout = []
    modify_scene_fns = []
    swing_peak = []
    linvel = []
    angvel = []

    x = -0.25
    command = jp.array([x, 0, 0])
    state = jit_reset(rng)
    for i in range(1_400):
        # Increase the forward velocity by 0.25 m/s every 200 steps.
        if i % 200 == 0:
            x += 0.25
            print(f"Setting x to {x}")
            command = jp.array([x, 0, 0])
        state.info["command"] = command
        if state.info["steps_since_last_pert"] < state.info["steps_until_next_pert"]:
            rng = sample_pert(rng)
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        rollout.append(state)
        swing_peak.append(state.info["swing_peak"])
        linvel.append(env.get_global_linvel(state.data))
        angvel.append(env.get_gyro(state.data))
        xyz = np.array(state.data.xpos[env._torso_body_id])
        xyz += np.array([0, 0, 0.2])
        x_axis = state.data.xmat[env._torso_body_id, 0]
        yaw = -np.arctan2(x_axis[1], x_axis[0])
        modify_scene_fns.append(
            functools.partial(
                draw_joystick_command,
                cmd=command,
                xyz=xyz,
                theta=yaw,
                scl=abs(command[0]) / env_cfg.command_config.a[0],
            )
        )


    # Plot each foot in a 2x2 grid.
    swing_peak = jp.array(swing_peak)
    names = ["FR", "FL", "RR", "RL"]
    colors = ["r", "g", "b", "y"]
    fig, axs = plt.subplots(2, 2)
    for i, ax in enumerate(axs.flat):
        ax.plot(swing_peak[:, i], color=colors[i])
        ax.set_ylim([0, env_cfg.reward_config.max_foot_height * 1.25])
        ax.axhline(env_cfg.reward_config.max_foot_height, color="k", linestyle="--")
        ax.set_title(names[i])
        ax.set_xlabel("time")
        ax.set_ylabel("height")
    plt.tight_layout()
    plt.show()

    linvel_x = jp.array(linvel)[:, 0]
    linvel_y = jp.array(linvel)[:, 1]
    angvel_yaw = jp.array(angvel)[:, 2]

    # Plot whether velocity is within the command range.
    linvel_x = jp.convolve(linvel_x, jp.ones(10) / 10, mode="same")
    linvel_y = jp.convolve(linvel_y, jp.ones(10) / 10, mode="same")
    angvel_yaw = jp.convolve(angvel_yaw, jp.ones(10) / 10, mode="same")

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    axes[0].plot(linvel_x)
    axes[1].plot(linvel_y)
    axes[2].plot(angvel_yaw)

    axes[0].set_ylim(
        -env_cfg.command_config.a[0], env_cfg.command_config.a[0]
    )
    axes[1].set_ylim(
        -env_cfg.command_config.a[1], env_cfg.command_config.a[1]
    )
    axes[2].set_ylim(
        -env_cfg.command_config.a[2], env_cfg.command_config.a[2]
    )

    for i, ax in enumerate(axes):
        ax.axhline(state.info["command"][i], color="red", linestyle="--")

    labels = ["dx", "dy", "dyaw"]
    for i, ax in enumerate(axes):
        ax.set_ylabel(labels[i])


    render_every = 2
    fps = 1.0 / eval_env.dt / render_every
    print(f"fps: {fps}")

    traj = rollout[::render_every]
    mod_fns = modify_scene_fns[::render_every]
    assert len(traj) == len(mod_fns)

    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[2] = True
    scene_option.geomgroup[3] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True

    frames = eval_env.render(
        traj,
        camera="track",
        height=480,
        width=640,
        modify_scene_fns=mod_fns,
        scene_option=scene_option,
    )
    media.show_video(frames, fps=fps, loop=False)
    saveVideo(frames)

def Handstand() :
    global env,env_cfg,env_name
    env_name = 'Go1Handstand'
    env = registry.load(env_name)
    env_cfg = registry.get_default_config(env_name)
    ppo_params = locomotion_params.brax_ppo_config(env_name)

    ckpt_path = epath.Path("checkpoints").resolve() / env_name
    ckpt_path.mkdir(parents=True, exist_ok=True)
    print(f"{ckpt_path}")

    with open(ckpt_path / "config.json", "w") as fp:
        json.dump(env_cfg.to_json(), fp, indent=4)

    x_data, y_data, y_dataerr = [], [], []
    times = [datetime.now()]


    def policy_params_fn(current_step, make_policy, params):
        del make_policy  # Unused.
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        path = ckpt_path / f"{current_step}"
        orbax_checkpointer.save(path, params, force=True, save_args=save_args)


    def progress(num_steps, metrics):
        clear_output(wait=True)

        times.append(datetime.now())
        x_data.append(num_steps)
        y_data.append(metrics["eval/episode_reward"])
        y_dataerr.append(metrics["eval/episode_reward_std"])

        plt.xlim([0, ppo_params["num_timesteps"] * 1.25])
        plt.xlabel("# environment steps")
        plt.ylabel("reward per episode")
        plt.title(f"y={y_data[-1]:.3f}")
        plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")

        # display(plt.gcf())
        display(plt)

    randomizer = registry.get_domain_randomizer(env_name)
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
        randomization_fn=randomizer,
        progress_fn=progress,
        policy_params_fn=policy_params_fn,
    )


    make_inference_fn, params, metrics = train_fn(
        environment=registry.load(env_name, config=env_cfg),
        eval_env=registry.load(env_name, config=env_cfg),
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )
    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")

    inference_fn = make_inference_fn(params, deterministic=True)
    jit_inference_fn = jax.jit(inference_fn)

    eval_env = registry.load(env_name, config=env_cfg)
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)

    rng = jax.random.PRNGKey(12345)
    rollout = []
    rewards = []
    torso_height = []
    actions = []
    torques = []
    power = []
    qfrc_constraint = []
    qvels = []
    power1 = []
    power2 = []
    for _ in range(10):
        rng, reset_rng = jax.random.split(rng)
        state = jit_reset(reset_rng)
        for i in range(env_cfg.episode_length // 2):
            act_rng, rng = jax.random.split(rng)
            ctrl, _ = jit_inference_fn(state.obs, act_rng)
            actions.append(ctrl)
            state = jit_step(state, ctrl)
            rollout.append(state)
            rewards.append(
                {k[7:]: v for k, v in state.metrics.items() if k.startswith("reward/")}
            )
            torso_height.append(state.data.qpos[2])
            torques.append(state.data.actuator_force)
            qvel = state.data.qvel[6:]
            power.append(jp.sum(jp.abs(qvel * state.data.actuator_force)))
            qfrc_constraint.append(jp.linalg.norm(state.data.qfrc_constraint[6:]))
            qvels.append(jp.max(jp.abs(qvel)))
            frc = state.data.actuator_force
            qvel = state.data.qvel[6:]
            power1.append(jp.sum(frc * qvel))
            power2.append(jp.sum(jp.abs(frc * qvel)))


    render_every = 2
    fps = 1.0 / eval_env.dt / render_every
    traj = rollout[::render_every]

    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[2] = True
    scene_option.geomgroup[3] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False

    frames = eval_env.render(
        traj, camera="side", scene_option=scene_option, height=480, width=640
    )
    media.show_video(frames, fps=fps, loop=False)
    saveVideo(frames)

    power = jp.array(power1)
    print(f"Max power: {jp.max(power)}")


    env_cfg = registry.get_default_config(env_name)
    env_cfg.energy_termination_threshold = 400  # lower energy termination threshold
    env_cfg.reward_config.energy = -0.003  # non-zero negative `energy` reward
    env_cfg.reward_config.dof_acc = -2.5e-7  # non-zero negative `dof_acc` reward

    FINETUNE_PATH = epath.Path(ckpt_path)
    latest_ckpts = list(FINETUNE_PATH.glob("*"))
    latest_ckpts = [ckpt for ckpt in latest_ckpts if ckpt.is_dir()]
    latest_ckpts.sort(key=lambda x: int(x.name))
    latest_ckpt = latest_ckpts[-1]
    restore_checkpoint_path = latest_ckpt

    x_data, y_data, y_dataerr = [], [], []
    times = [datetime.now()]

    make_inference_fn, params, metrics = train_fn(
        environment=registry.load(env_name, config=env_cfg),
        eval_env=registry.load(env_name, config=env_cfg),
        wrap_env_fn=wrapper.wrap_for_brax_training,
        restore_checkpoint_path=restore_checkpoint_path,  # restore from the checkpoint!
        seed=1,
    )
    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")

    inference_fn = make_inference_fn(params, deterministic=True)
    jit_inference_fn = jax.jit(inference_fn)

    eval_env = registry.load(env_name, config=env_cfg)
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)

    rng = jax.random.PRNGKey(12345)
    rollout = []
    rewards = []
    torso_height = []
    actions = []
    torques = []
    power = []
    qfrc_constraint = []
    qvels = []
    power1 = []
    power2 = []
    for _ in range(10):
        rng, reset_rng = jax.random.split(rng)
        state = jit_reset(reset_rng)
        for i in range(env_cfg.episode_length // 2):
            act_rng, rng = jax.random.split(rng)
            ctrl, _ = jit_inference_fn(state.obs, act_rng)
            actions.append(ctrl)
            state = jit_step(state, ctrl)
            rollout.append(state)
            rewards.append(
                {k[7:]: v for k, v in state.metrics.items() if k.startswith("reward/")}
            )
            torso_height.append(state.data.qpos[2])
            torques.append(state.data.actuator_force)
            qvel = state.data.qvel[6:]
            power.append(jp.sum(jp.abs(qvel * state.data.actuator_force)))
            qfrc_constraint.append(jp.linalg.norm(state.data.qfrc_constraint[6:]))
            qvels.append(jp.max(jp.abs(qvel)))
            frc = state.data.actuator_force
            qvel = state.data.qvel[6:]
            power1.append(jp.sum(frc * qvel))
            power2.append(jp.sum(jp.abs(frc * qvel)))


    render_every = 2
    fps = 1.0 / eval_env.dt / render_every
    traj = rollout[::render_every]

    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[2] = True
    scene_option.geomgroup[3] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False

    frames = eval_env.render(
        traj, camera="side", scene_option=scene_option, height=480, width=640
    )
    media.show_video(frames, fps=fps, loop=False)
    saveVideo(frames)

    power = jp.array(power1)
    print(f"Max power: {jp.max(power)}")

def Bipedal() :
    global env,env_cfg,env_name
    env_name = 'BerkeleyHumanoidJoystickFlatTerrain'
    env = registry.load(env_name)
    env_cfg = registry.get_default_config(env_name)
    ppo_params = locomotion_params.brax_ppo_config(env_name)


    x_data, y_data, y_dataerr = [], [], []
    times = [datetime.now()]

    def progress(num_steps, metrics):
        clear_output(wait=True)

        times.append(datetime.now())
        x_data.append(num_steps)
        y_data.append(metrics["eval/episode_reward"])
        y_dataerr.append(metrics["eval/episode_reward_std"])

        plt.xlim([0, ppo_params["num_timesteps"] * 1.25])
        plt.xlabel("# environment steps")
        plt.ylabel("reward per episode")
        plt.title(f"y={y_data[-1]:.3f}")
        plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")

        # display(plt.gcf())
        display(plt)

    randomizer = registry.get_domain_randomizer(env_name)
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
        randomization_fn=randomizer,
        progress_fn=progress
    )


    make_inference_fn, params, metrics = train_fn(
        environment=env,
        eval_env=registry.load(env_name, config=env_cfg),
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )
    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")

    env = registry.load(env_name)
    eval_env = registry.load(env_name)
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)
    jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

    rng = jax.random.PRNGKey(1)

    rollout = []
    modify_scene_fns = []

    x_vel = 1.0  #@param {type: "number"}
    y_vel = 0.0  #@param {type: "number"}
    yaw_vel = 0.0  #@param {type: "number"}
    command = jp.array([x_vel, y_vel, yaw_vel])

    phase_dt = 2 * jp.pi * eval_env.dt * 1.5
    phase = jp.array([0, jp.pi])

    for j in range(1):
        print(f"episode {j}")
        state = jit_reset(rng)
        state.info["phase_dt"] = phase_dt
        state.info["phase"] = phase
        for i in range(env_cfg.episode_length):
            act_rng, rng = jax.random.split(rng)
            ctrl, _ = jit_inference_fn(state.obs, act_rng)
            state = jit_step(state, ctrl)
            if state.done:
                break
            state.info["command"] = command
            rollout.append(state)

            xyz = np.array(state.data.xpos[eval_env.mj_model.body("torso").id])
            xyz += np.array([0, 0.0, 0])
            x_axis = state.data.xmat[eval_env._torso_body_id, 0]
            yaw = -np.arctan2(x_axis[1], x_axis[0])
            modify_scene_fns.append(
                functools.partial(
                    draw_joystick_command,
                    cmd=state.info["command"],
                    xyz=xyz,
                    theta=yaw,
                    scl=np.linalg.norm(state.info["command"]),
                )
        )

    render_every = 1
    fps = 1.0 / eval_env.dt / render_every
    print(f"fps: {fps}")
    traj = rollout[::render_every]
    mod_fns = modify_scene_fns[::render_every]

    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[2] = True
    scene_option.geomgroup[3] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False

    frames = eval_env.render(
        traj,
        camera="track",
        scene_option=scene_option,
        width=640*2,
        height=480,
        modify_scene_fns=mod_fns,
    )
    media.show_video(frames, fps=fps, loop=False)
    saveVideo(frames)

if __name__ == "__main__" :
    print("init_env")
    init_env()
    print("LoadQuadrupedalEnv")
    LoadQuadrupedalEnv()
    print("JoystickTrain")
    make_inference_fn,params = JoystickTrain()
    print("Rollout")
    Rollout(make_inference_fn,params)
    print("Handstand")
    Handstand()
    print("Bipedal")
    Bipedal()