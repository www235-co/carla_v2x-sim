from .client import Client
from .dataset import Dataset
import traceback

class Generator:
    def __init__(self,config):
        self.config = config
        self.collect_client = Client(self.config["client"])
        print('111',self.collect_client.client.get_available_maps())

    def generate_dataset(self,load=False):
        #初始化数据集（指定保存路径、版本，是否加载已有进度）
        self.dataset = Dataset(**self.config["dataset"],load=load)
        print("self.dataset.data",self.dataset.data["progress"])
        for sensor in self.config["sensors"]:
            self.dataset.update_sensor(sensor["name"],sensor["modality"])
        for category in self.config["categories"]:
            self.dataset.update_category(category["name"],category["description"])
        for attribute in self.config["attributes"]:
            self.dataset.update_attribute(attribute["name"],category["description"])
        for visibility in self.config["visibility"]:
            self.dataset.update_visibility(visibility["description"],visibility["level"])

        ## 循环生成每个世界（地图）的数据集
        print("self.config", self.config["worlds"])
        for world_config in self.config["worlds"][self.dataset.data["progress"]["current_world_index"]:]:
            try:
                self.collect_client.generate_world(world_config)# # 生成CARLA世界（加载地图、设置同步模式等）
                map_token = self.dataset.update_map(world_config["map_name"],world_config["map_category"])# 更新地图信息到数据集
                for capture_config in world_config["captures"][self.dataset.data["progress"]["current_capture_index"]:]:
                    log_token = self.dataset.update_log(map_token,capture_config["date"],capture_config["time"],
                                            capture_config["timezone"],capture_config["capture_vehicle"],capture_config["location"])
                    progress = self.dataset.data["progress"]
                    current_scene_idx = progress["current_scene_index"]
                    scenes_len = len(capture_config["scenes"])
                    print(f"current_scene_index: {current_scene_idx}, scenes长度: {scenes_len}")
                    # 循环处理每个采集配置（log级）
                    for scene_config in capture_config["scenes"][self.dataset.data["progress"]["current_scene_index"]:]:

                        print("scene_config", scene_config)
                        # 循环生成 scene_config["count"] 次场景（这里是 1 次）
                        for scene_count in range(self.dataset.data["progress"]["current_scene_count"],scene_config["count"]):
                            self.dataset.update_scene_count()
                            self.add_one_scene(log_token,scene_config)
                            self.dataset.save()
                        self.dataset.update_scene_index()
                    self.dataset.update_capture_index()
                self.dataset.update_world_index()
            except:
                traceback.print_exc()
            finally:
                self.collect_client.destroy_world()
                
    def add_one_scene(self,log_token,scene_config):
        try:
            calibrated_sensors_token = {}
            samples_data_token = {}
            instances_token = {}
            samples_annotation_token = {}

            self.collect_client.generate_scene(scene_config)
            scene_token = self.dataset.update_scene(log_token,scene_config["description"])
            print("scene_token",scene_token)

            for instance in self.collect_client.walkers+self.collect_client.vehicles:
                # 获取实例（车辆/行人）的基本信息，生成唯一标识 instance_token
                instance_token = self.dataset.update_instance(*self.collect_client.get_instance(scene_token,instance))
                # 用 Carla 内部的 actor ID 作为键，存储实例标识（便于后续帧中快速查找）
                instances_token[instance.get_actor().id] = instance_token
                # 初始化标注标识（后续关键帧中会更新为实际标注的 token）
                samples_annotation_token[instance.get_actor().id] = ""
            
            for sensor in self.collect_client.sensors:
                # 获取传感器的校准参数，生成唯一标识 calibrated_sensor_token
                calibrated_sensor_token = self.dataset.update_calibrated_sensor(scene_token,*self.collect_client.get_calibrated_sensor(sensor))
                # 用传感器名称作为键，存储校准标识（便于后续关联传感器数据）
                calibrated_sensors_token[sensor.name] = calibrated_sensor_token
                # 初始化传感器数据标识（后续关键帧中会更新为实际数据的 token）
                samples_data_token[sensor.name] = ""

            sample_token = ""   # 关键帧的唯一标识（初始为空，第一帧会生成）
            # 计算总帧数：场景采集时间 ÷ 模拟器帧间隔（固定为 0.01 秒）
            # 例如：collect_time=1 秒 → 1 / 0.01 = 100 帧
            #按模拟器的最小时间单位（帧）循环推进场景，确保所有动态变化（车辆移动、传感器数据生成）被逐帧捕获。
            for frame_count in range(int(scene_config["collect_time"]/self.collect_client.settings.fixed_delta_seconds)):
                print("frame count:",frame_count)
                self.collect_client.tick()## 触发 Carla 模拟器更新一帧
                # 推进模拟器时间（前进 fixed_delta_seconds 秒，即 0.01 秒）。
                # 更新所有实体的状态：车辆按轨迹移动、行人行走、主车行驶。
                # 触发传感器（相机、激光雷达等）生成当前帧的原始数据（如 RGB 图像、点云），并缓存到 sensor.get_data_list() 中。

                # 计算关键帧间隔帧数：keyframe_time ÷ 帧间隔 → 例如 0.5 秒 / 0.01 秒 = 50 帧
                if (frame_count+1)%int(scene_config["keyframe_time"]/self.collect_client.settings.fixed_delta_seconds) == 0:
                    i = 0
                    i += 1
                    print(f"关键帧，保存数据的帧{i}")
                    sample_token = self.dataset.update_sample(sample_token,scene_token,*self.collect_client.get_sample())# 更新关键帧信息，生成唯一标识 sample_token
                    # 遍历所有传感器（只处理指定类型：相机、雷达、激光雷达）
                    for sensor in self.collect_client.sensors:
                        if sensor.bp_name in ['sensor.camera.rgb','sensor.other.radar','sensor.lidar.ray_cast']:
                            # 遍历传感器在当前帧缓存的所有数据（可能有多帧，如雷达可能一次返回多段数据）
                            for idx,sample_data in enumerate(sensor.get_data_list()):
                                # 1. 记录主车在该传感器数据采集时的位姿（位置+朝向）
                                ego_pose_token = self.dataset.update_ego_pose(scene_token,calibrated_sensors_token[sensor.name],*self.collect_client.get_ego_pose(sample_data))
                                is_key_frame = False # 2. 标记是否为该传感器在当前关键帧的最后一段数据
                                if idx == len(sensor.get_data_list())-1:
                                    is_key_frame = True# 最后一段数据标记为关键帧（用于后续数据关联）
                                # 3. 保存传感器数据到数据集
                                samples_data_token[sensor.name] = self.dataset.update_sample_data(samples_data_token[sensor.name],calibrated_sensors_token[sensor.name],sample_token,ego_pose_token,is_key_frame,*self.collect_client.get_sample_data(sample_data))
                    # 遍历所有车辆和行人
                    for instance in self.collect_client.walkers+self.collect_client.vehicles:
                        # 只处理可见的实体（在传感器视野内，无遮挡或部分遮挡）
                        if self.collect_client.get_visibility(instance) > 0:
                            # 更新该实体在当前关键帧的标注信息
                            samples_annotation_token[instance.get_actor().id]  = self.dataset.update_sample_annotation(samples_annotation_token[instance.get_actor().id],sample_token,*self.collect_client.get_sample_annotation(scene_token,instance))
                    for sensor in self.collect_client.sensors:
                        sensor.get_data_list().clear()
        except:
            traceback.print_exc()
        finally:
            self.collect_client.destroy_scene()