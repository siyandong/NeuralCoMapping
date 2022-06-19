# Parts of the code in this file have been borrowed from:
#    https://github.com/facebookresearch/habitat-api

import numpy as np
import torch
import atexit
import multiprocessing
import sys
import traceback

from .exploration_env import Exploration_Env


class SyncNavEnv():
    def __init__(self, env_constructors, device):
        """Batch together environments and simulate them in external processes.
        The environments can be different but must use the same action and
        observation specs.

        :param env_constructors: List of callables that create environments.
        :param blocking: Whether to step environments one after another.
        :param flatten: Boolean, whether to use flatten action and time_steps during
            communication to reduce overhead.
        :raise ValueError: If the action or observation specs don't match.
        """
        self._envs = [ctor() for ctor in env_constructors]
        self._num_envs = len(env_constructors)
        self.action_space = self._envs[0].action_space
        self.observation_space = self._envs[0].observation_space
        self._device = device

    def reset(self):
        """Reset all environments and combine the resulting observation.

        :return: a list of [next_obs, reward, done, info]
        """
        for env in self._envs:
            env.reset()

    def step(self, actions):
        """Forward a batch of actions to the wrapped environments.

        :param actions: batched action, possibly nested, to apply to the environment.
        :return: a list of [next_obs, reward, done, info]
        """
        done = [env.step(actions[i].numpy()) for i, env in enumerate(self._envs)]
        return done

    def get_short_term_goal(self, inputs, visuals):
        time_steps = [env.get_short_term_goal(input, visual) for env, input, visual in zip(self._envs, inputs, visuals)]
        stg = np.vstack(time_steps)
        stg = torch.from_numpy(stg).float()
        return stg

    def get_camera_info(self, idx=-1):
        camera_info = [env.get_camera_info(idx) for i, env in enumerate(self._envs)]
        return camera_info

class ParallelNavEnv():
    """Batch together environments and simulate them in external processes.
    The environments are created in external processes by calling the provided
    callables. This can be an environment class, or a function creating the
    environment and potentially wrapping it. The returned environment should not
    access global variables.
    """

    def __init__(self, env_constructors, device, blocking=False, flatten=False):
        """Batch together environments and simulate them in external processes.
        The environments can be different but must use the same action and
        observation specs.

        :param env_constructors: List of callables that create environments.
        :param blocking: Whether to step environments one after another.
        :param flatten: Boolean, whether to use flatten action and time_steps during
            communication to reduce overhead.
        :raise ValueError: If the action or observation specs don't match.
        """
        self._envs = [ProcessPyEnvironment(
            ctor, flatten=flatten) for ctor in env_constructors]
        self._num_envs = len(env_constructors)
        self.start()
        self.action_space = self._envs[0].action_space
        self.observation_space = self._envs[0].observation_space
        self._blocking = blocking
        self._flatten = flatten
        self._device = device

    def start(self):
        """
        Start all children processes
        """
        for env in self._envs:
            env.start()

    @property
    def batched(self):
        return True

    @property
    def batch_size(self):
        return self._num_envs

    def reset(self):
        """Reset all environments and combine the resulting observation.

        :return: a list of [next_obs, reward, done, info]
        """
        time_steps = [env.apply('reset', (), self._blocking) for env in self._envs]
        if not self._blocking:
            time_steps = [promise() for promise in time_steps]

    def step(self, actions):
        """Forward a batch of actions to the wrapped environments.

        :param actions: batched action, possibly nested, to apply to the environment.
        :return: a list of [next_obs, reward, done, info]
        """
        done = [env.apply('step', (actions[i].numpy(),), self._blocking)
                      for i, env in enumerate(self._envs)]
        # When blocking is False we get promises that need to be called.
        if not self._blocking:
            done = [promise() for promise in done]
        return done

    def get_short_term_goal(self, inputs, visuals):
        time_steps = [env.apply('get_short_term_goal', (input, visual), self._blocking)
                      for env, input, visual in zip(self._envs, inputs, visuals)]
        if not self._blocking:
            time_steps = [promise() for promise in time_steps]
        stg = np.vstack(time_steps)
        stg = torch.from_numpy(stg).float()
        return stg

    def close(self):
        """Close all external process."""
        for env in self._envs:
            env.close()

class ProcessPyEnvironment(object):
    """Step a single env in a separate process for lock free paralellism."""

    # Message types for communication via the pipe.
    _READY = 1
    _ACCESS = 2
    _CALL = 3
    _RESULT = 4
    _EXCEPTION = 5
    _CLOSE = 6

    def __init__(self, env_constructor, flatten=False):
        """Step environment in a separate process for lock free paralellism.

        The environment is created in an external process by calling the provided
        callable. This can be an environment class, or a function creating the
        environment and potentially wrapping it. The returned environment should
        not access global variables.

        :param env_constructor: callable that creates and returns a Python environment.
        :param flatten: boolean, whether to assume flattened actions and time_steps
        during communication to avoid overhead.
        """
        self._env_constructor = env_constructor
        self._flatten = flatten

    def start(self):
        """Start the process."""
        self._conn, conn = multiprocessing.Pipe()
        self._process = multiprocessing.Process(target=self._worker,
                                                args=(conn, self._env_constructor, self._flatten))
        atexit.register(self.close)
        self._process.start()
        result = self._conn.recv()
        if isinstance(result, Exception):
            self._conn.close()
            self._process.join(5)
            raise result
        assert result is self._READY, result

    def __getattr__(self, name):
        """Request an attribute from the environment.
        Note that this involves communication with the external process, so it can
        be slow.

        :param name: attribute to access.
        :return: value of the attribute.
        """
        self._conn.send((self._ACCESS, name))
        return self._receive()

    def call(self, name, *args, **kwargs):
        """Asynchronously call a method of the external environment.

        :param name: name of the method to call.
        :param args: positional arguments to forward to the method.
        :param kwargs: keyword arguments to forward to the method.
        :return: promise object that blocks and provides the return value when called.
        """
        payload = name, args, kwargs
        self._conn.send((self._CALL, payload))
        return self._receive

    def close(self):
        """Send a close message to the external process and join it."""
        try:
            self._conn.send((self._CLOSE, None))
            self._conn.close()
        except IOError:
            # The connection was already closed.
            pass
        self._process.join(5)

    def apply(self, func, param = [], blocking=True):
        """Step the environment.

        :param action: the action to apply to the environment.
        :param blocking: whether to wait for the result.
        :return: (next_obs, reward, done, info) tuple when blocking, otherwise callable that returns that tuple
        """
        promise = self.call(func, *param)
        if blocking:
            return promise()
        else:
            return promise

    def _receive(self):
        """Wait for a message from the worker process and return its payload.

        :raise Exception: an exception was raised inside the worker process.
        :raise KeyError: the reveived message is of an unknown type.

        :return: payload object of the message.
        """
        message, payload = self._conn.recv()
        # Re-raise exceptions in the main process.
        if message == self._EXCEPTION:
            stacktrace = payload
            raise Exception(stacktrace)
        if message == self._RESULT:
            return payload
        self.close()
        raise KeyError(
            'Received message of unexpected type {}'.format(message))

    def _worker(self, conn, env_constructor, flatten=False):
        """The process waits for actions and sends back environment results.

        :param conn: connection for communication to the main process.
        :param env_constructor: env_constructor for the OpenAI Gym environment.
        :param flatten: boolean, whether to assume flattened actions and
        time_steps during communication to avoid overhead.

        :raise KeyError: when receiving a message of unknown type.
        """
        try:
            np.random.seed()
            env = env_constructor()
            conn.send(self._READY)    # Ready.
            while True:
                try:
                    # Only block for short times to have keyboard exceptions be raised.
                    if not conn.poll(0.1):
                        continue
                    message, payload = conn.recv()
                except (EOFError, KeyboardInterrupt):
                    break
                if message == self._ACCESS:
                    name = payload
                    result = getattr(env, name)
                    conn.send((self._RESULT, result))
                    continue
                if message == self._CALL:
                    name, args, kwargs = payload
                    # if name in ['step', 'reset', 'get_short_term_goal']:
                    result = getattr(env, name)(*args, **kwargs)
                    conn.send((self._RESULT, result))
                    continue
                if message == self._CLOSE:
                    assert payload is None
                    break
                raise KeyError(
                    'Received message of unknown type {}'.format(message))
        except Exception:    # pylint: disable=broad-except
            etype, evalue, tb = sys.exc_info()
            stacktrace = ''.join(traceback.format_exception(etype, evalue, tb))
            message = 'Error in environment process: {}'.format(stacktrace)
            # tf.logging.error(message)
            conn.send((self._EXCEPTION, stacktrace))
        finally:
            conn.close()


def make_env_fn_binded(args, config_file, scene_id, device_idx, rank, snp_data):
    def make_env_fn():
        print("Loading {}".format(config_file))
        env = Exploration_Env(args, config_file, scene_id, device_idx, rank, snp_data)
        return env
    return make_env_fn


def construct_envs(args, snp_data):
    with open(args.scenes_file, 'r') as f:
        scenes = []
        for idx, x in enumerate(f.readlines()):
            if idx % args.scene_per_process == 0:
                scenes.append([])
            scenes[-1].append(x.rstrip())
    
    make_env_fns = []
    for i, scene_ids in enumerate(scenes):
        gpu_id = args.use_single_gpu
        make_env_fns.append(make_env_fn_binded(args, args.config_file, scene_ids, gpu_id, i, snp_data))

    if len(scenes) == 1:
        envs = SyncNavEnv(make_env_fns, args.device)
    else:
        envs = ParallelNavEnv(make_env_fns, args.device)

    return envs
