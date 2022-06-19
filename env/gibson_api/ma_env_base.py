from gibson2.robots.turtlebot_robot import Turtlebot
from gibson2.robots.husky_robot import Husky
from gibson2.robots.ant_robot import Ant
from gibson2.robots.humanoid_robot import Humanoid
from gibson2.robots.jr2_robot import JR2
from gibson2.robots.jr2_kinova_robot import JR2_Kinova
from gibson2.robots.freight_robot import Freight
from gibson2.robots.fetch_robot import Fetch
from gibson2.robots.locobot_robot import Locobot
from gibson2.simulator import Simulator
from gibson2.scenes.empty_scene import EmptyScene
from gibson2.scenes.stadium_scene import StadiumScene
from gibson2.scenes.gibson_indoor_scene import StaticIndoorScene
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.utils.utils import parse_config
import gibson2.render.mesh_renderer as mesh_renderer
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
from gibson2.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings

from gibson2.utils.utils import quatToXYZW
from gibson2.tasks.room_rearrangement_task import RoomRearrangementTask
from gibson2.tasks.point_nav_fixed_task import PointNavFixedTask
from gibson2.tasks.point_nav_random_task import PointNavRandomTask
from gibson2.tasks.interactive_nav_random_task import InteractiveNavRandomTask
from gibson2.tasks.dynamic_nav_random_task import DynamicNavRandomTask
from gibson2.tasks.reaching_random_task import ReachingRandomTask
from .sensors import VisionSensor, BumpSensor
from gibson2.robots.robot_base import BaseRobot
from gibson2.external.pybullet_tools.utils import stable_z_on_aabb

from transforms3d.euler import euler2quat
from collections import OrderedDict
import skimage.morphology
import argparse
import gym
import numpy as np
import pybullet as p
import time
import os
import logging

# multi-agent version

g_frag_shader = ('''
#version 450
uniform sampler2D texUnit;
uniform sampler2D metallicTexture;
uniform sampler2D roughnessTexture;
uniform sampler2D normalTexture;
uniform samplerCube specularTexture;
uniform samplerCube irradianceTexture;
uniform sampler2D specularBRDF_LUT;
uniform samplerCube specularTexture2;
uniform samplerCube irradianceTexture2;
uniform sampler2D specularBRDF_LUT2;
uniform sampler2D lightModulationMap;
uniform vec3 eyePosition;
uniform float use_texture;
uniform float use_pbr;
uniform float use_pbr_mapping;
uniform float use_two_light_probe;
uniform float metallic;
uniform float roughness;
uniform sampler2D depthMap;
uniform int shadow_pass;
uniform vec3 light_position;  // in world coordinate
uniform vec3 light_color;

in vec2 theCoords;
in vec3 Normal_world;
in vec3 Normal_cam;
in vec3 FragPos;
in vec3 Instance_color;
in vec3 Pos_cam;
in vec3 Pos_cam_prev;
in vec3 Pos_cam_projected;
in vec3 Diffuse_color;
in mat3 TBN;
in vec4 FragPosLightSpace;
in vec2 Optical_flow;

layout (location = 0) out vec4 outputColour;
layout (location = 1) out vec4 NormalColour;
layout (location = 2) out vec4 InstanceColour;
layout (location = 3) out vec4 PCColour;
layout (location = 4) out vec4 SceneFlowColour;
layout (location = 5) out vec4 OpticalFlowColour;

void main() {
    vec3 lightDir = vec3(0,0,1); //sunlight pointing to z direction
    float diff = 0.5 + 0.5 * max(dot(Normal_world, lightDir), 0.0);
    vec3 diffuse = diff * light_color;
    vec2 texelSize = 1.0 / textureSize(depthMap, 0);
    float shadow;
    if (shadow_pass == 2) {
        vec3 projCoords = FragPosLightSpace.xyz / FragPosLightSpace.w;
        projCoords = projCoords * 0.5 + 0.5;
        float cosTheta = dot(Normal_world, lightDir);
        cosTheta = clamp(cosTheta, 0.0, 1.0);
        float bias = 0.005*tan(acos(cosTheta));
        bias = clamp(bias, 0.001 ,0.1);
        float currentDepth = projCoords.z;
        float closestDepth = 0;
        shadow = 0.0;
        float current_shadow = 0;
        for(int x = -2; x <= 2; ++x) {
            for (int y = -2; y <= 2; ++y) {
                closestDepth = texture(depthMap, projCoords.xy + vec2(x, y) * texelSize).b * 0.5 + 0.5;
                current_shadow = currentDepth - bias > closestDepth  ? 1.0 : 0.0;
                if ((projCoords.z > 1.0) || (projCoords.x > 1.0) || (projCoords.y > 1.0) || (projCoords.x < 0) || (projCoords.y < 0)) current_shadow = 0.0;
                shadow += current_shadow;
            }
        }
        shadow /= 25.0;
    } else shadow = 0.0;
    if (use_texture == 1) outputColour = texture(texUnit, theCoords);// albedo only
    else outputColour = vec4(Diffuse_color,1) * diff; //diffuse color
    NormalColour =  vec4((Normal_cam + 1) / 2,1);
    InstanceColour = vec4(1, 1, 1, 1);
    if (shadow_pass == 1) PCColour = vec4(Pos_cam_projected, 1);
    else PCColour = vec4(Pos_cam, 1);
    outputColour = outputColour *  (1 - shadow * 0.5);
    SceneFlowColour =  vec4(Pos_cam - Pos_cam_prev,1);
    OpticalFlowColour =  vec4(Optical_flow,0,1);
}
''', '''
#version 450
uniform sampler2D texUnit;
uniform sampler2D metallicTexture;
uniform sampler2D roughnessTexture;
uniform sampler2D normalTexture;
uniform samplerCube specularTexture;
uniform samplerCube irradianceTexture;
uniform sampler2D specularBRDF_LUT;
uniform samplerCube specularTexture2;
uniform samplerCube irradianceTexture2;
uniform sampler2D specularBRDF_LUT2;
uniform sampler2D lightModulationMap;
uniform vec3 eyePosition;
uniform float use_texture;
uniform float use_pbr;
uniform float use_pbr_mapping;
uniform float use_two_light_probe;
uniform float metallic;
uniform float roughness;
uniform sampler2D depthMap;
uniform int shadow_pass;
uniform vec3 light_position;  // in world coordinate
uniform vec3 light_color;

in vec2 theCoords;
in vec3 Normal_world;
in vec3 Normal_cam;
in vec3 FragPos;
in vec3 Instance_color;
in vec3 Pos_cam;
in vec3 Pos_cam_prev;
in vec3 Pos_cam_projected;
in vec3 Diffuse_color;
in mat3 TBN;
in vec4 FragPosLightSpace;
in vec2 Optical_flow;

layout (location = 0) out vec4 outputColour;
layout (location = 1) out vec4 NormalColour;
layout (location = 2) out vec4 InstanceColour;
layout (location = 3) out vec4 PCColour;
layout (location = 4) out vec4 SceneFlowColour;
layout (location = 5) out vec4 OpticalFlowColour;

void main() {
    outputColour = vec4(1, 1, 1, 1);
    NormalColour =  vec4((Normal_cam + 1) / 2, 1);
    InstanceColour = vec4(1, 1, 1, 1);
    if (shadow_pass == 1) PCColour = vec4(Pos_cam_projected, 1);
    else PCColour = vec4(Pos_cam, 1);
    SceneFlowColour =  vec4(Pos_cam - Pos_cam_prev, 1);
    OpticalFlowColour =  vec4(Optical_flow,0,1);
}
''')

class SimulatorEx(Simulator):
    def __init__(
        self,
        gravity=9.8,
        physics_timestep=1 / 120.0,
        render_timestep=1 / 30.0,
        mode='gui',
        image_width=128,
        image_height=128,
        vertical_fov=90,
        device_idx=0,
        render_to_tensor=False,
        rendering_settings=MeshRendererSettings(),
        need_rgb=True
    ):
        assert not rendering_settings.optimized
        assert not render_to_tensor
        assert not rendering_settings.enable_pbr
        assert not rendering_settings.msaa
        self.need_rgb = need_rgb
        super(SimulatorEx, self).__init__(
            gravity=gravity,
            physics_timestep=physics_timestep,
            render_timestep=render_timestep,
            mode=mode,
            image_width=image_width,
            image_height=image_height,
            vertical_fov=vertical_fov,
            device_idx=device_idx,
            render_to_tensor=render_to_tensor,
            rendering_settings=rendering_settings
        )
        assert not self.use_ig_renderer

    def load(self):
        self.renderer = MeshRenderer(
            width=self.image_width,
            height=self.image_height,
            vertical_fov=self.vertical_fov,
            device_idx=self.device_idx,
            rendering_settings=self.rendering_settings
        )

        assert self.renderer.platform not in ['Darwin', 'Windows']
        
        global g_frag_shader
        if self.need_rgb:
            frag_shader = g_frag_shader[0]
        else:
            frag_shader = g_frag_shader[1]

        self.renderer.shaderProgram = self.renderer.r.compile_shader_meshrenderer(
            "".join(open(os.path.join(
                    os.path.dirname(mesh_renderer.__file__),
                    'shaders',
                    '450',
                    'vert.shader')
                ).readlines()),
            frag_shader
        )

        if self.use_pb_renderer:
            self.cid = p.connect(p.GUI)
        else:
            self.cid = p.connect(p.DIRECT)
        p.setTimeStep(self.physics_timestep)
        p.setGravity(0, 0, -self.gravity)
        p.setPhysicsEngineParameter(enableFileCaching=0)

        self.visual_objects = {}
        self.robots = []
        self.scene = None
    

class BaseEnv(gym.Env):
    '''
    Base Env class, follows OpenAI Gym interface
    Handles loading scene and robot
    Functions like reset and step are not implemented
    '''

    def __init__(self,
                 config_file,
                 scene_ids,
                 mode='headless',
                 action_timestep=1 / 10.0,
                 physics_timestep=1 / 240.0,
                 render_to_tensor=False,
                 device_idx=0):
        """
        :param config_file: config_file path
        :param scene_id: override scene_id in config file
        :param mode: headless or gui mode
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param device_idx: device_idx: which GPU to run the simulation and rendering on
        """
        self.config = parse_config(config_file)
        self.scene_list = scene_ids
        self.config['scene_id'] = self.scene_list[0]

        self.num_robots = self.config['num_robots']
        self.reset_orientation = self.config['reset_orientation']
        self.reset_floor = self.config['reset_floor']
        self.mode = mode
        self.action_timestep = action_timestep
        self.physics_timestep = physics_timestep
        self.texture_randomization_freq = self.config.get(
            'texture_randomization_freq', None)
        self.object_randomization_freq = self.config.get(
            'object_randomization_freq', None)
        self.object_randomization_idx = 0
        self.num_object_randomization_idx = 10

        enable_shadow = self.config.get('enable_shadow', False)
        enable_pbr = self.config.get('enable_pbr', False)
        texture_scale = self.config.get('texture_scale', 1.0)
        need_rgb = 'rgb' in self.config['output']

        settings = MeshRendererSettings(enable_shadow=enable_shadow,
                                        enable_pbr=enable_pbr,
                                        msaa=False,
                                        texture_scale=texture_scale)

        self.simulator = SimulatorEx(mode=mode,
                                   physics_timestep=physics_timestep,
                                   render_timestep=action_timestep,
                                   image_width=self.config.get(
                                       'image_width', 128),
                                   image_height=self.config.get(
                                       'image_height', 128),
                                   vertical_fov=self.config.get(
                                       'vertical_fov', 90),
                                   device_idx=device_idx,
                                   render_to_tensor=render_to_tensor,
                                   rendering_settings=settings,
                                   need_rgb=need_rgb)
        self.load()

    def reload(self, config_file):
        """
        Reload another config file
        Thhis allows one to change the configuration on the fly

        :param config_file: new config file path
        """
        self.config = parse_config(config_file)
        self.simulator.reload()
        self.load()

    def reload_model(self, scene_id):
        """
        Reload another scene model
        This allows one to change the scene on the fly

        :param scene_id: new scene_id
        """
        if self.config['scene_id'] != scene_id:
            self.config['scene_id'] = scene_id
            self.simulator.reload()
            self.load()

    def reload_model_object_randomization(self):
        """
        Reload the same model, with the next object randomization random seed
        """
        if self.object_randomization_freq is None:
            return
        self.object_randomization_idx = (self.object_randomization_idx + 1) % \
            (self.num_object_randomization_idx)
        self.simulator.reload()
        self.load()

    def get_next_scene_random_seed(self):
        """
        Get the next scene random seed
        """
        if self.object_randomization_freq is None:
            return None
        return self.scene_random_seeds[self.scene_random_seed_idx]

    def load(self):
        """
        Load the scene and robot
        """
        if self.config['scene'] == 'empty':
            scene = EmptyScene()
            self.simulator.import_scene(
                scene, load_texture=self.config.get('load_texture', True))
        elif self.config['scene'] == 'stadium':
            scene = StadiumScene()
            self.simulator.import_scene(
                scene, load_texture=self.config.get('load_texture', True))
        elif self.config['scene'] == 'gibson':
            scene = StaticIndoorScene(
                self.config['scene_id'],
                waypoint_resolution=self.config.get(
                    'waypoint_resolution', 0.2),
                num_waypoints=self.config.get('num_waypoints', 10),
                build_graph=self.config.get('build_graph', False),
                trav_map_resolution=self.config.get(
                    'trav_unit_size_cm', 0.1),
                trav_map_erosion=self.config.get('trav_map_erosion', 2),
                pybullet_load_texture=self.config.get(
                    'pybullet_load_texture', False) and self.config['scene_id'] not in ['CVPR2022', 'Cross'],
            )
            self.simulator.import_scene(
                scene, load_texture=self.config.get('load_texture', True) and self.config['scene_id'] not in ['CVPR2022', 'Cross'])
        elif self.config['scene'] == 'igibson':
            scene = InteractiveIndoorScene(
                self.config['scene_id'],
                waypoint_resolution=self.config.get(
                    'waypoint_resolution', 0.2),
                num_waypoints=self.config.get('num_waypoints', 10),
                build_graph=self.config.get('build_graph', False),
                trav_map_resolution=self.config.get(
                    'trav_unit_size_cm', 0.1),
                trav_map_erosion=self.config.get('trav_map_erosion', 2),
                trav_map_type=self.config.get('trav_map_type', 'with_obj'),
                pybullet_load_texture=self.config.get(
                    'pybullet_load_texture', False),
                texture_randomization=self.texture_randomization_freq is not None,
                object_randomization=self.object_randomization_freq is not None,
                object_randomization_idx=self.object_randomization_idx,
                should_open_all_doors=self.config.get(
                    'should_open_all_doors', False),
                load_object_categories=self.config.get(
                    'load_object_categories', None),
                load_room_types=self.config.get('load_room_types', None),
                load_room_instances=self.config.get(
                    'load_room_instances', None),
            )
            # TODO: Unify the function import_scene and take out of the if-else clauses
            first_n = self.config.get('_set_first_n_objects', -1)
            if first_n != -1:
                scene._set_first_n_objects(first_n)
            self.simulator.import_ig_scene(scene)

        if self.config['robot'] == 'Turtlebot':
            robot = Turtlebot
        elif self.config['robot'] == 'Husky':
            robot = Husky
        elif self.config['robot'] == 'Ant':
            robot = Ant
        elif self.config['robot'] == 'Humanoid':
            robot = Humanoid
        elif self.config['robot'] == 'JR2':
            robot = JR2
        elif self.config['robot'] == 'JR2_Kinova':
            robot = JR2_Kinova
        elif self.config['robot'] == 'Freight':
            robot = Freight
        elif self.config['robot'] == 'Fetch':
            robot = Fetch
        elif self.config['robot'] == 'Locobot':
            robot = Locobot
        else:
            raise Exception(
                'unknown robot type: {}'.format(self.config['robot']))

        self.scene = scene
        self.robots = [robot(self.config) for i in range(self.num_robots)]
        for robot in self.robots:
            self.simulator.import_robot(robot)
            p.changeDynamics(robot.robot_ids[0], -1, lateralFriction=0.)
        p.changeDynamics(self.scene.mesh_body_id, -1, lateralFriction=0.)

    def clean(self):
        """
        Clean up
        """
        if self.simulator is not None:
            self.simulator.disconnect()

    def close(self):
        """
        Synonymous function with clean
        """
        self.clean()

    def simulator_step(self):
        """
        Step the simulation.
        This is different from environment step that returns the next
        observation, reward, done, info.
        """
        self.simulator.step()

    def step(self, action):
        """
        Overwritten by subclasses
        """
        return NotImplementedError()

    def reset(self):
        """
        Overwritten by subclasses
        """
        return NotImplementedError()

    def set_mode(self, mode):
        """
        Set simulator mode
        """
        self.simulator.mode = mode



class MAGibsonEnv(BaseEnv):
    """
    Multi-Agent iGibson Environment (OpenAI Gym interface)
    """

    def __init__(
        self,
        config_file,
        scene_ids=None,
        mode='headless',
        action_timestep=1 / 5.0,
        physics_timestep=1 / 80.0,
        device_idx=0,
        render_to_tensor=False,
        automatic_reset=False,
    ):
        """
        :param config_file: config_file path
        :param scene_ids: override scene_ids in config file
        :param mode: headless, gui, iggui
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param device_idx: which GPU to run the simulation and rendering on
        :param render_to_tensor: whether to render directly to pytorch tensors
        :param automatic_reset: whether to automatic reset after an episode finishes
        """
        super(MAGibsonEnv, self).__init__(config_file=config_file,
                                          scene_ids=scene_ids,
                                          mode=mode,
                                          action_timestep=action_timestep,
                                          physics_timestep=physics_timestep,
                                          device_idx=device_idx,
                                          render_to_tensor=render_to_tensor)
        self.automatic_reset = automatic_reset
        self.base_mask = None

    def load_task_setup(self):
        """
        Load task setup
        """
        self.initial_pos_z_offset = self.config.get(
            'initial_pos_z_offset', 0.1)
        # s = 0.5 * G * (t ** 2)
        drop_distance = 0.5 * 9.8 * (self.action_timestep ** 2)
        assert drop_distance < self.initial_pos_z_offset, \
            'initial_pos_z_offset is too small for collision checking'

        # ignore the agent's collision with these body ids
        self.collision_ignore_body_b_ids = set(
            self.config.get('collision_ignore_body_b_ids', []))
        # ignore the agent's collision with these link ids of itself
        self.collision_ignore_link_a_ids = set(
            self.config.get('collision_ignore_link_a_ids', []))

        # discount factor
        self.discount_factor = self.config.get('discount_factor', 0.99)

        # domain randomization frequency
        self.texture_randomization_freq = self.config.get(
            'texture_randomization_freq', None)
        self.object_randomization_freq = self.config.get(
            'object_randomization_freq', None)

        # task
        self.task = self.config.get('task', None)
        if self.task is not None:
            raise NotImplementedError

    def build_obs_space(self, shape, low, high):
        """
        Helper function that builds individual observation spaces
        """
        return gym.spaces.Box(
            low=low,
            high=high,
            shape=shape,
            dtype=np.float32)

    def load_observation_space(self):
        """
        Load observation space
        """
        self.output = self.config['output']
        self.image_width = self.config.get('image_width', 128)
        self.image_height = self.config.get('image_height', 128)
        observation_space = OrderedDict()
        sensors = OrderedDict()
        vision_modalities = []
        scan_modalities = []

        if 'task_obs' in self.output:
            raise NotImplementedError
        if 'rgb' in self.output:
            observation_space['rgb'] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3),
                low=0.0, high=1.0)
            vision_modalities.append('rgb')
        if 'depth' in self.output:
            observation_space['depth'] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 1),
                low=0.0, high=1.0)
            vision_modalities.append('depth')
        if 'pc' in self.output:
            observation_space['pc'] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3),
                low=-np.inf, high=np.inf)
            vision_modalities.append('pc')
        if 'optical_flow' in self.output:
            observation_space['optical_flow'] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 2),
                low=-np.inf, high=np.inf)
            vision_modalities.append('optical_flow')
        if 'scene_flow' in self.output:
            observation_space['scene_flow'] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3),
                low=-np.inf, high=np.inf)
            vision_modalities.append('scene_flow')
        if 'normal' in self.output:
            observation_space['normal'] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3),
                low=-np.inf, high=np.inf)
            vision_modalities.append('normal')
        if 'seg' in self.output:
            observation_space['seg'] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 1),
                low=0.0, high=1.0)
            vision_modalities.append('seg')
        if 'rgb_filled' in self.output:  # use filler
            observation_space['rgb_filled'] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3),
                low=0.0, high=1.0)
            vision_modalities.append('rgb_filled')
        if 'label' in self.output:
            observation_space['label'] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 1),
                low=0.0, high=1.0)
            vision_modalities.append('label')
        if 'scan' in self.output:
            raise NotImplementedError
        if 'occupancy_grid' in self.output:
            raise NotImplementedError
        if 'bump' in self.output:
            observation_space['bump'] = gym.spaces.Box(low=0.0,
                                                       high=1.0,
                                                       shape=(1,))
            sensors['bump'] = BumpSensor(self)

        if len(vision_modalities) > 0:
            sensors['vision'] = VisionSensor(self, vision_modalities)

        if len(scan_modalities) > 0:
            raise NotImplementedError

        self.observation_space = gym.spaces.Dict(observation_space)
        self.sensors = sensors

    def load_action_space(self):
        """
        Load action space
        """
        self.action_space = self.robots[0].action_space

    def load_miscellaneous_variables(self):
        """
        Load miscellaneous variables for book keeping
        """
        self.current_step = 0
        self.collision_step = 0
        self.current_episode = 0
        self.collision_links = [[] for _ in range(self.num_robots)]

    def load(self):
        """
        Load environment
        """
        super(MAGibsonEnv, self).load()
        self.load_task_setup()
        self.load_observation_space()
        self.load_action_space()
        self.load_miscellaneous_variables()

    def get_state(self, collision_links=[]):
        """
        Get the current observation

        :param collision_links: collisions from last physics timestep
        :return: observation as a dictionary
        """
        state = OrderedDict()
        if 'task_obs' in self.output:
            raise NotImplementedError
        if 'vision' in self.sensors:
            vision_obs = self.sensors['vision'].get_obs(self)
            for modality in vision_obs:
                state[modality] = np.stack(vision_obs[modality])
        if 'scan_occ' in self.sensors:
            raise NotImplementedError
        if 'bump' in self.sensors:
            state['bump'] = np.array(self.sensors['bump'].get_obs(self))

        return state

    def run_simulation(self):
        """
        Run simulation for one action timestep (same as one render timestep in Simulator class)

        :return: collision_links: collisions from last physics timestep
        """
        self.simulator_step()
        collision_links = [list(p.getContactPoints(bodyA=self.robots[na].robot_ids[0])) for na in range(self.num_robots)]
        return self.filter_collision_links(collision_links)

    def filter_collision_links(self, collision_links):
        """
        Filter out collisions that should be ignored

        :param collision_links: original collisions, a list of collisions
        :return: filtered collisions
        """
        full_links = []
        for na in range(self.num_robots):
            new_collision_links = []
            for item in collision_links[na]:
                # ignore collision with body b
                if item[2] in self.collision_ignore_body_b_ids:
                    continue

                # ignore collision with robot link a
                if item[3] in self.collision_ignore_link_a_ids:
                    continue

                # ignore self collision with robot link a (body b is also robot itself)
                if item[2] == self.robots[0].robot_ids[0] and item[4] in self.collision_ignore_link_a_ids:
                    continue
                new_collision_links.append(item)
            full_links.append(new_collision_links)
        return full_links

    def populate_info(self, info):
        """
        Populate info dictionary with any useful information
        """
        info['episode_length'] = self.current_step
        info['collision_step'] = self.collision_step

    def step(self, actions):
        """
        Apply robot's action.
        Returns the next state, reward, done and info,
        following OpenAI Gym's convention

        :param action: robot actions
        :return: state: next observation
        :return: reward: reward of this time step
        :return: done: whether the episode is terminated
        :return: info: info dictionary with any useful information
        """
        self.current_step += 1
        if actions is not None:
            for i in range(self.num_robots):
                self.robots[i].apply_action(actions[i])
        collision_links = self.run_simulation()
        self.collision_links = collision_links
        self.collision_step += int(sum(map(len, collision_links)) > 0)

        state = self.get_state(collision_links)
        info = {}
        done = False
        reward = 0.

        if self.task is not None:
            reward, info = self.task.get_reward(
                self, collision_links, action, info)
            done, info = self.task.get_termination(
                self, collision_links, action, info)
            self.task.step(self)
        self.populate_info(info)

        if done and self.automatic_reset:
            info['last_observation'] = state
            state = self.reset()

        return state, reward, done, info

    def check_collision(self, body_id):
        """
        Check with the given body_id has any collision after one simulator step

        :param body_id: pybullet body id
        :return: whether the given body_id has no collision
        """
        self.simulator_step()
        collisions = list(p.getContactPoints(bodyA=body_id))

        if logging.root.level <= logging.DEBUG:  # Only going into this if it is for logging --> efficiency
            for item in collisions:
                logging.debug('bodyA:{}, bodyB:{}, linkA:{}, linkB:{}'.format(
                    item[1], item[2], item[3], item[4]))

        return len(collisions) == 0

    def check_collision_between(self, a, b):
        return p.getContactPoints(bodyA=self.robots[a].robot_ids[0], bodyB=self.robots[b].robot_ids[0])

    def set_pos_orn_with_z_offset(self, obj, pos, orn=None, offset=None):
        """
        Reset position and orientation for the robot or the object

        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        :param offset: z offset
        """
        if orn is None:
            orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])

        if offset is None:
            offset = self.initial_pos_z_offset

        is_robot = isinstance(obj, BaseRobot)
        body_id = obj.robot_ids[0] if is_robot else obj.body_id
        # first set the correct orientation
        obj.set_position_orientation(pos, quatToXYZW(euler2quat(*orn), 'wxyz'))
        # compute stable z based on this orientation
        stable_z = stable_z_on_aabb(body_id, [pos, pos])
        # change the z-value of position with stable_z + additional offset
        # in case the surface is not perfect smooth (has bumps)
        obj.set_position([pos[0], pos[1], stable_z + offset])

    def test_valid_position(self, obj, pos, orn=None):
        """
        Test if the robot or the object can be placed with no collision

        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        :return: validity
        """
        is_robot = isinstance(obj, BaseRobot)

        self.set_pos_orn_with_z_offset(obj, pos, orn)

        if is_robot:
            obj.robot_specific_reset()
            obj.keep_still()

        body_id = obj.robot_ids[0] if is_robot else obj.body_id
        has_collision = self.check_collision(body_id)
        return has_collision

    def land(self, obj, pos, orn):
        """
        Land the robot or the object onto the floor, given a valid position and orientation

        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        """
        is_robot = isinstance(obj, BaseRobot)

        self.set_pos_orn_with_z_offset(obj, pos, orn)

        if is_robot:
            obj.robot_specific_reset()
            obj.keep_still()

        body_id = obj.robot_ids[0] if is_robot else obj.body_id

        land_success = False
        # land for maximum 1 second, should fall down ~5 meters
        max_simulator_step = int(1.0 / self.action_timestep)
        for _ in range(max_simulator_step):
            self.simulator_step()
            if len(p.getContactPoints(bodyA=body_id)) > 0:
                land_success = True
                break

        if not land_success:
            logging.warning("WARNING: Failed to land")

        if is_robot:
            obj.robot_specific_reset()
            obj.keep_still()

    def reset_variables(self):
        """
        Reset bookkeeping variables for the next new episode
        """
        self.current_episode += 1
        self.current_step = 0
        self.collision_step = 0
        self.collision_links = [[] for _ in range(self.num_robots)]

    def randomize_domain(self):
        """
        Domain randomization
        Object randomization loads new object models with the same poses
        Texture randomization loads new materials and textures for the same object models
        """
        if self.object_randomization_freq is not None:
            if self.current_episode % self.object_randomization_freq == 0:
                self.reload_model_object_randomization()
        if self.texture_randomization_freq is not None:
            if self.current_episode % self.texture_randomization_freq == 0:
                self.simulator.scene.randomize_texture()

    def base_point_mask(self, x, y, shape):
        if self.base_mask is None or shape[0] != self.base_mask.shape[0] or shape[1] != self.base_mask.shape[1]:
            mask = np.zeros((shape[0] * 2 + 1, shape[1] * 2 + 1), dtype=np.int32)
            mask[shape[0], shape[1]] = 1
            self.base_mask = (skimage.morphology.binary_dilation(mask, skimage.morphology.disk(self.config['reset_max_dist'])) ^ skimage.morphology.binary_dilation(mask, skimage.morphology.disk(self.config['reset_min_dist']))) > 0
        return self.base_mask[shape[0]-x:2*shape[0]-x, shape[1]-y:2*shape[1]-y]
        

    def get_random_point_near(self, floor, base_point):
        trav = self.scene.floor_map[floor] == 255
        if base_point is not None:
            base_point = self.scene.world_to_map(base_point)
            bx, by = base_point[0], base_point[1]
            trav = np.logical_and(trav, self.base_point_mask(bx, by, trav.shape))
        trav_space = np.where(trav)
        if trav_space[0].shape[0] == 0:
            return None, None
        idx = np.random.randint(0, high=trav_space[0].shape[0])
        xy_map = np.array([trav_space[0][idx], trav_space[1][idx]])
        x, y = self.scene.map_to_world(xy_map)
        z = self.scene.floor_heights[floor]
        return floor, np.array([x, y, z])

    
    def reset(self):
        if len(self.scene_list) > 1 and self.current_episode > 0:
            current_episode = self.current_episode
            self.reload_model(self.scene_list[current_episode % len(self.scene_list)])
            self.current_episode = current_episode
        """
        Reset episode
        """
        max_reset = 50
        for i in range(max_reset):
            self.randomize_domain()
            # move robot away from the scene
            for i in range(self.num_robots):
                self.robots[i].set_position([100.0 * (i + 1), 100.0 * (i + 1), 100.0 * (i + 1)])
            if self.task is not None:
                self.task.reset_scene(self)
                self.task.reset_agent(self)
                break
            else:
                
                floor_num = self.scene.get_random_floor() if self.reset_floor else 0
                self.scene.reset_floor(floor=floor_num, additional_elevation=0.07)
                
                reset_success = False
                max_trials = 20 * self.num_robots if self.num_robots < 50 else 4 * self.num_robots
                initial_pos = []
                orn = np.array([0, 0, (np.random.uniform(0, np.pi * 2) if self.reset_orientation else 0)])
                idx_agent = 0
                last_pos = None if self.config['scene_id'] not in ['CVPR2022', 'Cross'] else np.array([0, 0])
                # cache pybullet state
                # TODO: p.saveState takes a few seconds, need to speed up
                # state_id = p.saveState()
                for i in range(max_trials):
                    _, pos = self.get_random_point_near(floor=floor_num, base_point=last_pos)
                    if pos is None:
                        idx_agent = 0
                        last_pos = None if self.config['scene_id'] not in ['CVPR2022', 'Cross'] else np.array([0, 0])
                        continue
                    reset_success = self.test_valid_position(self.robots[idx_agent], pos, orn)
                    # p.restoreState(state_id)
                    if reset_success:
                        initial_pos.append(pos)
                        last_pos = pos[:2]
                        orn = np.array([0, 0, (np.random.uniform(0, np.pi * 2) if self.reset_orientation else 0)])
                        idx_agent += 1
                        if idx_agent == self.num_robots:
                            break

                close_assumption = True
                for i in range(len(initial_pos)):
                    for j in range(i + 1, len(initial_pos)):
                        if (((initial_pos[i] - initial_pos[i + 1])[:2] ** 2).sum() > 3 ** 2):
                            close_assumption = False
                assert close_assumption

                if idx_agent == self.num_robots:
                    break

        if idx_agent < self.num_robots:
            logging.error("ERROR: Failed to reset robot without collision")
            raise RuntimeError("Failed to reset robot without collision")

        # p.removeState(state_id)

        # land for maximum 1 second, should fall down ~5 meters
        max_simulator_step = int(1.0 / self.action_timestep)
        for _ in range(max_simulator_step):
            self.simulator_step()

        if sum([len(p.getContactPoints(bodyA=obj.robot_ids[0])) for obj in self.robots]) == 0:
            logging.warning("WARNING: Failed to land")

        for obj in self.robots:
            obj.robot_specific_reset()
            obj.keep_still()

        self.simulator.sync()
        state = self.get_state()
        self.reset_variables()

        return state, floor_num