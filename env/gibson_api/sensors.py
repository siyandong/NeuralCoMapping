from gibson2.sensors.sensor_base import BaseSensor
from gibson2.sensors.sensor_noise_base import BaseSensorNoise
from gibson2.render.mesh_renderer.instances import Robot

import numpy as np
import os
import gibson2
from gibson2.utils.mesh_util import xyzw2wxyz, quat2rotmat
from collections import OrderedDict
from gibson2.utils.constants import AVAILABLE_MODALITIES, ShadowPass

class BumpSensor(BaseSensor):
    """
    Bump sensor
    """

    def __init__(self, env):
        super(BumpSensor, self).__init__(env)

    def get_obs(self, env):
        """
        Get Bump sensor reading

        :return: Bump sensor reading
        """
        has_collision = [float(len(links) > 0) for links in env.collision_links]
        return has_collision

class DropoutSensorNoise(BaseSensorNoise):
    """
    Naive dropout sensor noise model
    """

    def __init__(self, env):
        super(DropoutSensorNoise, self).__init__(env)
        self.noise_rate = 0.0
        self.noise_value = 1.0
        self.rng = np.random.RandomState(np.random.randint(0, 65536))

    def set_noise_rate(self, noise_rate):
        """
        Set noise rate

        :param noise_rate: noise rate
        """
        self.noise_rate = noise_rate

    def set_noise_value(self, noise_value):
        """
        Set noise value

        :param noise_value: noise value
        """
        self.noise_value = noise_value

    def add_noise(self, obs):
        """
        Add naive sensor dropout to perceptual sensor, such as RGBD and LiDAR scan

        :param sensor_reading: raw sensor reading, range must be between [0.0, 1.0]
        :param noise_rate: how much noise to inject, 0.05 means 5% of the data will be replaced with noise_value
        :param noise_value: noise_value to overwrite raw sensor reading
        :return: sensor reading corrupted with noise
        """
        if self.noise_rate <= 0.0:
            return obs

        assert len(obs[(obs < 0.0) | (obs > 1.0)]) == 0,\
            'sensor reading has to be between [0.0, 1.0]'

        valid_mask = self.rng.choice(2, obs.shape, p=[
                                      self.noise_rate, 1.0 - self.noise_rate])
        obs[valid_mask == 0] = self.noise_value
        return obs



class VisionSensor(BaseSensor):
    """
    Vision sensor (including rgb, rgb_filled, depth, 3d, seg, normal, optical flow, scene flow)
    """

    def __init__(self, env, modalities):
        super(VisionSensor, self).__init__(env)
        self.modalities = modalities
        self.raw_modalities = self.get_raw_modalities(modalities)
        self.image_width = self.config.get('image_width', 128)
        self.image_height = self.config.get('image_height', 128)

        self.depth_noise_rate = self.config.get('depth_noise_rate', 0.0)
        self.depth_low = self.config.get('depth_low', 0.5)
        self.depth_high = self.config.get('depth_high', 5.0)

        self.noise_model = DropoutSensorNoise(env)
        self.noise_model.set_noise_rate(self.depth_noise_rate)
        self.noise_model.set_noise_value(0.0)

        if 'rgb_filled' in modalities:
            try:
                import torch.nn as nn
                import torch
                from torchvision import transforms
                from gibson2.learn.completion import CompletionNet
            except ImportError:
                raise Exception(
                    'Trying to use rgb_filled ("the goggle"), but torch is not installed. Try "pip install torch torchvision".')

            self.comp = CompletionNet(norm=nn.BatchNorm2d, nf=64)
            self.comp = torch.nn.DataParallel(self.comp).cuda()
            self.comp.load_state_dict(torch.load(
                os.path.join(gibson2.assets_path, 'networks', 'model.pth')))
            self.comp.eval()

    def get_raw_modalities(self, modalities):
        """
        Helper function that gathers raw modalities (e.g. depth is based on 3d)

        :return: raw modalities to query the renderer
        """
        raw_modalities = []
        if 'rgb' in modalities or 'rgb_filled' in modalities:
            raw_modalities.append('rgb')
        if 'depth' in modalities or 'pc' in modalities:
            raw_modalities.append('3d')
        if 'seg' in modalities:
            raw_modalities.append('seg')
        if 'normal' in modalities:
            raw_modalities.append('normal')
        if 'optical_flow' in modalities:
            raw_modalities.append('optical_flow')
        if 'scene_flow' in modalities:
            raw_modalities.append('scene_flow')
        return raw_modalities

    def get_rgb(self, raw_vision_obs):
        """
        :return: RGB sensor reading, normalized to [0.0, 1.0]
        """
        return raw_vision_obs['rgb'][:, :, :3]

    def get_rgb_filled(self, raw_vision_obs):
        """
        :return: RGB-filled sensor reading by passing through the "Goggle" neural network
        """
        rgb = self.get_rgb(raw_vision_obs)
        with torch.no_grad():
            tensor = transforms.ToTensor()((rgb * 255).astype(np.uint8)).cuda()
            rgb_filled = self.comp(tensor[None, :, :, :])[0]
            return rgb_filled.permute(1, 2, 0).cpu().numpy()

    def get_depth(self, raw_vision_obs):
        """
        :return: depth sensor reading, normalized to [0.0, 1.0]
        """
        depth = -raw_vision_obs['3d'][:, :, 2:3]
        # 0.0 is a special value for invalid entries
        depth[depth < self.depth_low] = 0.0
        depth[depth > self.depth_high] = 0.0

        # re-scale depth to [0.0, 1.0]
        depth /= self.depth_high
        depth = self.noise_model.add_noise(depth)

        return depth

    def get_pc(self, raw_vision_obs):
        """
        :return: pointcloud sensor reading
        """
        return raw_vision_obs['3d'][:, :, :3]

    def get_optical_flow(self, raw_vision_obs):
        """
        :return: optical flow sensor reading
        """
        return raw_vision_obs['optical_flow'][:, :, :3]

    def get_scene_flow(self, raw_vision_obs):
        """
        :return: scene flow sensor reading
        """
        return raw_vision_obs['scene_flow'][:, :, :3]

    def get_normal(self, raw_vision_obs):
        """
        :return: surface normal reading
        """
        return raw_vision_obs['normal'][:, :, :3]

    def get_seg(self, raw_vision_obs):
        """
        :return: semantic segmentation mask, normalized to [0.0, 1.0]
        """
        seg = raw_vision_obs['seg'][:, :, 0:1]
        return seg

    def get_obs(self, env):
        """
        Get vision sensor reading

        :return: vision sensor reading
        """
        raw_vision_obs, label_vision_obs = self.get_render_results(env.simulator.renderer, modes=self.raw_modalities, need_label='label' in self.modalities)

        len_modalities = len(self.raw_modalities)

        raw_vision_obs = [{
            mode: raw_vision_obs[a * len_modalities + idx]
            for idx, mode in enumerate(self.raw_modalities)
        } for a in range(env.num_robots)]

        vision_obs = OrderedDict()
        if 'rgb' in self.modalities:
            vision_obs['rgb'] = [self.get_rgb(obs) for obs in raw_vision_obs]
        if 'rgb_filled' in self.modalities:
            vision_obs['rgb_filled'] = [self.get_rgb_filled(obs) for obs in raw_vision_obs]
        if 'depth' in self.modalities:
            vision_obs['depth'] = [self.get_depth(obs) for obs in raw_vision_obs]
        if 'pc' in self.modalities:
            vision_obs['pc'] = [self.get_pc(obs) for obs in raw_vision_obs]
        if 'optical_flow' in self.modalities:
            vision_obs['optical_flow'] = [self.get_optical_flow(obs) for obs in raw_vision_obs]
        if 'scene_flow' in self.modalities:
            vision_obs['scene_flow'] = [self.get_scene_flow(obs) for obs in raw_vision_obs]
        if 'normal' in self.modalities:
            vision_obs['normal'] = [self.get_normal(obs) for obs in raw_vision_obs]
        if 'seg' in self.modalities:
            vision_obs['seg'] = [self.get_seg(obs) for obs in raw_vision_obs]

        if 'label' in self.modalities:
            vision_obs['label'] = [obs[:, :, 0:1] for obs in label_vision_obs]

        return vision_obs


    def get_render_results(self, renderer, modes, need_label):
        frames = []
        labels = []
        non_robots = [instance for instance in renderer.instances if not isinstance(instance, Robot)]
        robots = [instance for instance in renderer.instances if isinstance(instance, Robot)]
        for robot in robots:
            camera_pos = robot.robot.eyes.get_position()
            orn = robot.robot.eyes.get_orientation()
            mat = quat2rotmat(xyzw2wxyz(orn))[:3, :3]
            view_direction = mat.dot(np.array([1, 0, 0]))
            renderer.set_camera(camera_pos, camera_pos + view_direction, [0, 0, 1], cache=True)

            render_shadow_pass = 'rgb' in modes
            need_flow_info = 'optical_flow' in modes or 'scene_flow' in modes
            renderer.update_dynamic_positions(need_flow_info=need_flow_info)

            if renderer.enable_shadow and render_shadow_pass:
                # shadow pass
                renderer.r.render_meshrenderer_pre(0, 0, renderer.fbo)
                
                for instance in renderer.instances:
                    if instance.shadow_caster:
                        instance.render(shadow_pass=ShadowPass.HAS_SHADOW_RENDER_SHADOW)

                renderer.r.render_meshrenderer_post()

                renderer.r.readbuffer_meshrenderer_shadow_depth(
                    renderer.width, renderer.height, renderer.fbo, renderer.depth_tex_shadow)

            # main pass
            renderer.r.render_meshrenderer_pre(0, 0, renderer.fbo)
            for instance in robots:
                if instance is robot:
                    continue
                if renderer.enable_shadow:
                    instance.render(
                        shadow_pass=ShadowPass.HAS_SHADOW_RENDER_SCENE)
                else:
                    instance.render(
                        shadow_pass=ShadowPass.NO_SHADOW)
            if need_label:
                for frame in renderer.readbuffer(['seg']):
                    labels.append(frame)
            for instance in non_robots:
                if renderer.enable_shadow:
                    instance.render(
                        shadow_pass=ShadowPass.HAS_SHADOW_RENDER_SCENE)
                else:
                    instance.render(
                        shadow_pass=ShadowPass.NO_SHADOW)

            renderer.r.render_meshrenderer_post()
            for frame in renderer.readbuffer(modes):
                frames.append(frame)
        return frames, labels
