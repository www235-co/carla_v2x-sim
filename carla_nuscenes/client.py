import carla
from .sensor import *
from .vehicle import Vehicle
from .walker import Walker
import math
from .utils import generate_token,get_nuscenes_rt,get_intrinsic,transform_timestamp,clamp
import random
import logging

class Client:
    def __init__(self,client_config):
        self.client = carla.Client(client_config["host"],client_config["port"])
        self.client.set_timeout(client_config["time_out"])# 设置连接超时时间

    def generate_world(self,world_config):
        print("generate world start!")
        self.client.load_world(world_config["map_name"])# 加载配置中指定的地图（如 "Town05_Opt"）
        self.world = self.client.get_world() # 获取 Carla 世界对象（核心交互接口）
        self.original_settings = self.world.get_settings()# 保存世界原始设置（用于后续恢复）
        self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)# 卸载地图中的静态停放车辆（避免干扰自定义场景的实体布局）
        self.ego_vehicle = None # 初始化实体容器（后续会存储主车、传感器、其他车辆、行人）
        self.sensors = None
        self.vehicles = None
        self.walkers = None

        # 定义匿名函数：根据蓝图 ID 判断实体类别
        get_category = lambda bp: "vehicle.car" if bp.id.split(".")[0] == "vehicle" else "human.pedestrian.adult" if bp.id.split(".")[0] == "walker" else None
        # 生成类别字典：{蓝图 ID: 类别}（用于快速查询实体类型）
        self.category_dict = {bp.id: get_category(bp) for bp in self.world.get_blueprint_library()}
        # 定义匿名函数：根据蓝图 ID 生成实体属性
        get_attribute = lambda bp: ["vehicle.moving"] if bp.id.split(".")[0] == "vehicle" else ["pedestrian.moving"] if bp.id.split(".")[0] == "walker" else None
        # 生成属性字典：{蓝图 ID: 属性列表}（用于标注实体动态特征）
        self.attribute_dict = {bp.id: get_attribute(bp) for bp in self.world.get_blueprint_library()}

        self.trafficmanager = self.client.get_trafficmanager()# 获取交通管理器（控制车辆自动驾驶行为的模块）
        self.trafficmanager.set_global_distance_to_leading_vehicle(1.0)
        self.trafficmanager.set_synchronous_mode(True) # 启用同步模式（与模拟器帧同步，确保数据一致性）
        self.trafficmanager.set_hybrid_physics_mode(True)
        self.trafficmanager.set_hybrid_physics_radius(100)
        self.trafficmanager.set_respawn_dormant_vehicles(True) # 自动重新激活静止车辆（避免道路空驶）
        self.trafficmanager.set_boundaries_respawn_dormant_vehicles(21, 70)

        self.settings = carla.WorldSettings(**world_config["settings"])# 应用世界运行参数（从配置文件读取，如帧间隔 fixed_delta_seconds=0.01）
        self.settings.synchronous_mode = True # 强制启用同步模式（关键！确保传感器数据与实体状态严格对应）
        self.settings.no_rendering_mode = False# 关闭无渲染模式（否则传感器无法生成图像/点云）
        self.world.apply_settings(self.settings)
        self.world.set_pedestrians_cross_factor(1)# 设置行人过马路概率为 100%（确保场景中行人行为更真实）
        print("generate world success!")

    def generate_scene(self,scene_config):
        print("generate scene start!")
        if scene_config["custom"]:
            # 生成自定义场景（完全按配置文件参数生成）
            self.generate_custom_scene(scene_config)
            print("生成自定义场景")
        else:
            # 生成随机场景（参数随机，用于快速扩充数据集多样性）
            self.generate_random_scene(scene_config)
        print("generate scene success!")

    def generate_custom_scene(self, scene_config):

        if scene_config["weather_mode"] == "custom":  # 根据配置选择天气模式（自定义参数或预设模式）
            self.weather = carla.WeatherParameters(**scene_config["weather"])
        else:
            self.weather = getattr(carla.WeatherParameters, scene_config["weather_mode"])

        self.world.set_weather(self.weather)  # 应用天气设置
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # --------------------------
        # 自车（Ego Vehicle）生成逻辑
        # --------------------------
        # 1. 从配置文件读取自车基础参数（车型、初始位置、旋转角），但忽略路径
        ego_config = scene_config["ego_vehicle"]
        self.ego_vehicle = Vehicle(world=self.world, **ego_config)  # 仍使用Vehicle类初始化
        print("self.ego_vehicle",self.ego_vehicle)
        self.ego_vehicle.blueprint.set_attribute('role_name', 'hero')  # 保留主车标记
        self.ego_vehicle.spawn_actor()  #  在 Carla 世界中生成主车实体
        self.ego_vehicle.get_actor().set_autopilot()  # 4. 启用主车自动驾驶

        # --------------------------
        # 环境车辆生成（核心修改部分）
        # --------------------------
        FILTERV = "vehicle.*"
        self.vehicles = []

        #  筛选有效随机生成点（原始范围）
        spawn_points_all = self.world.get_map().get_spawn_points()
        spawn_points = [
            t for t in spawn_points_all
            if -200 < t.location.x < 200 and -200 < t.location.y < 200
        ]
        number_of_spawn_points = len(spawn_points)
        if number_of_spawn_points == 0:
            logging.warning("未找到有效生成点，无法生成环境车辆")
            return
        # 确定生成数量
        NUM_OF_VEHICLES = scene_config.get("num_vehicles", 80)
        NUM_OF_VEHICLES = min(NUM_OF_VEHICLES, number_of_spawn_points)
        random.shuffle(spawn_points)

        # 3. 筛选车辆蓝图
        blueprints = self.world.get_blueprint_library().filter(FILTERV)
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        blueprints = [x for x in blueprints if not x.id.endswith(('isetta', 'carlacola', 'cybertruck', 't2'))]
        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        # 4. 生成车辆（添加碰撞检测与重试）
        spawned_count = 0  # 已成功生成的车辆数
        retry_limit = 3  # 每个点的最大重试次数
        used_spawn_points = set()  # 记录已使用的生成点（避免重复）

        while spawned_count < NUM_OF_VEHICLES and len(used_spawn_points) < number_of_spawn_points:
            # 选择未使用的生成点
            for i in range(len(spawn_points)):
                if i in used_spawn_points:
                    continue
                transform = spawn_points[i]
                used_spawn_points.add(i)  # 标记为已尝试

                # 随机选择蓝图
                blueprint = random.choice(blueprints)
                if blueprint.has_attribute('color'):
                    blueprint.set_attribute('color', random.choice(blueprint.get_attribute('color').recommended_values))
                if blueprint.has_attribute('driver_id'):
                    blueprint.set_attribute('driver_id',
                                            random.choice(blueprint.get_attribute('driver_id').recommended_values))
                blueprint.set_attribute('role_name', 'autopilot')

                # 尝试生成（带重试机制）
                for retry in range(retry_limit):
                    try:
                        # 执行单次生成（非批量，便于检测单个点的碰撞）
                        actor_id = self.world.spawn_actor(blueprint, transform).id
                        # 生成成功
                        vehicle_actor = self.world.get_actor(actor_id)
                        vehicle_actor.set_autopilot(True, self.trafficmanager.get_port())

                        # 创建Vehicle实例
                        vehicle = Vehicle(
                            world=self.world,
                            bp_name=blueprint.id,
                            location={
                                "x": transform.location.x,
                                "y": transform.location.y,
                                "z": transform.location.z
                            },
                            rotation={
                                "pitch": transform.rotation.pitch,
                                "yaw": transform.rotation.yaw,
                                "roll": transform.rotation.roll
                            }
                        )
                        vehicle.set_actor(actor_id)
                        self.vehicles.append(vehicle)
                        spawned_count += 1
                        logging.info(f"成功生成车辆（{spawned_count}/{NUM_OF_VEHICLES}），ID: {actor_id}")
                        break  # 跳出重试循环

                    except Exception as e:
                        # 捕获碰撞错误，重试其他位置
                        if "collision" in str(e).lower():
                            print(f"生成点[{i}]碰撞，重试第{retry + 1}/{retry_limit}次...")
                            if retry == retry_limit - 1:
                                logging.error(f"生成点[{i}]超过最大重试次数，放弃该点")
                        else:
                            print(f"生成失败：{str(e)}")
                            break  # 非碰撞错误，无需重试

                if spawned_count >= NUM_OF_VEHICLES:
                    break  # 达到目标数量，停止生成

        # 最终生成结果
        print(f"环境车辆生成完成，共成功生成 {len(self.vehicles)}/{NUM_OF_VEHICLES} 辆")

        # 行人生成逻辑（改进版）
        FILTERW = "walker.pedestrian.*"
        self.walkers = []
        self.actor_list = []
        self.non_player = []

        NUM_OF_WALKERS = scene_config.get("num_walkers", 8)
        if NUM_OF_WALKERS <= 0:
            logging.info("未配置行人数量，不生成行人")
        else:
            blueprintsWalkers = self.world.get_blueprint_library().filter(FILTERW)
            percentagePedestriansRunning = 0.0
            percentagePedestriansCrossing = 0.0

            # 生成行人spawn points（扩大范围并增加重试）
            spawn_points = []
            max_attempts = NUM_OF_WALKERS * 10  # 增加尝试次数
            attempt = 0

            # 扩大行人坐标范围，增加找到有效点的概率
            x_min, x_max = -200, 200  # 扩大x范围
            y_min, y_max = -200, 200  # 扩大y范围

            def is_safe_spawn(blueprint, transform):
                try:
                    temp_actor = self.world.spawn_actor(blueprint, transform)
                    temp_actor.destroy()
                    return True
                except Exception as e:
                    return "collision" not in str(e).lower()

            while len(spawn_points) < NUM_OF_WALKERS and attempt < max_attempts:
                spawn_point = carla.Transform()
                loc = self.world.get_random_location_from_navigation()

                # 放宽位置限制
                while loc is None or not (x_min < loc.x < x_max and y_min < loc.y < y_max):
                    loc = self.world.get_random_location_from_navigation()
                    attempt += 1
                    if attempt >= max_attempts:
                        break

                if loc is not None and attempt < max_attempts:
                    spawn_point.location = loc
                    # 尝试微调位置以避免碰撞
                    found = False
                    for _ in range(3):  # 微调尝试
                        if is_safe_spawn(blueprintsWalkers[0], spawn_point):
                            spawn_points.append(spawn_point)
                            found = True
                            break
                        # 微调位置
                        dx = random.uniform(-0.5, 0.5)
                        dy = random.uniform(-0.5, 0.5)
                        spawn_point.location.x += dx
                        spawn_point.location.y += dy

                    if not found:
                        logging.warning(f"尝试位置 {spawn_point.location} 存在碰撞，已跳过")

                attempt += 1

            # 处理生成点结果
            actual_num_walkers = len(spawn_points)
            if actual_num_walkers < NUM_OF_WALKERS:
                logging.warning(f"仅找到 {actual_num_walkers} 个安全的行人生成点，少于请求的 {NUM_OF_WALKERS} 个")
                NUM_OF_WALKERS = actual_num_walkers

            if NUM_OF_WALKERS > 0:
                # 生成行人实体
                batch = []
                walker_speed = []
                for spawn_point in spawn_points:
                    walker_bp = random.choice(blueprintsWalkers)
                    if walker_bp.has_attribute('is_invincible'):
                        walker_bp.set_attribute('is_invincible', 'false')
                    if walker_bp.has_attribute('speed'):
                        if random.random() > percentagePedestriansRunning:
                            walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                        else:
                            walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
                    else:
                        walker_speed.append(0.0)
                    batch.append(SpawnActor(walker_bp, spawn_point))

                results = self.client.apply_batch_sync(batch, True)
                walker_speed2 = []
                walkers_list = []
                for i in range(len(results)):
                    if results[i].error:
                        logging.error(results[i].error)
                    else:
                        walkers_list.append({"id": results[i].actor_id})
                        walker_speed2.append(walker_speed[i])
                walker_speed = walker_speed2

                if walkers_list:
                    # 生成行人控制器
                    batch = []
                    walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
                    for i in range(len(walkers_list)):
                        batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
                    results = self.client.apply_batch_sync(batch, True)

                    for i in range(len(results)):
                        if not results[i].error:
                            walkers_list[i]["con"] = results[i].actor_id

                    # 收集有效实体
                    all_id = []
                    walkers_id = []
                    for i in range(len(walkers_list)):
                        if "con" in walkers_list[i]:
                            all_id.append(walkers_list[i]["con"])
                            all_id.append(walkers_list[i]["id"])
                            walkers_id.append(walkers_list[i]["id"])

                    all_actors = self.world.get_actors(all_id)
                    walker_actors = self.world.get_actors(walkers_id)
                    self.non_player.extend(walker_actors)
                    self.actor_list.extend(all_actors)

                    self.world.tick()

                    # 初始化控制器
                    self.world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
                    for i in range(0, len(all_id), 2):
                        try:
                            all_actors[i].start()
                            all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
                            all_actors[i].set_max_speed(float(walker_speed[int(i / 2)]))
                        except Exception as e:
                            logging.error(f"初始化行人控制器失败: {str(e)}")

                            # 9. 创建Walker实例（修复旋转属性处理）
                            for walker in walkers_list:
                                if "con" not in walker:
                                    continue
                                try:
                                    # 获取Carla原生行人实体
                                    carla_actor = self.world.get_actor(walker["id"])
                                    # 直接获取原生实体的旋转信息，不依赖Walker类的方法
                                    carla_rotation = carla_actor.get_rotation()

                                    # 创建Walker对象时传入原始旋转数据
                                    walker_obj = Walker(
                                        world=self.world,
                                        bp_name=carla_actor.type_id,
                                        location={
                                            "x": carla_actor.get_location().x,
                                            "y": carla_actor.get_location().y,
                                            "z": carla_actor.get_location().z
                                        },
                                        rotation={
                                            "pitch": carla_rotation.pitch,
                                            "yaw": carla_rotation.yaw,
                                            "roll": carla_rotation.roll
                                        }
                                    )
                                    walker_obj.set_actor(walker["id"])
                                    walker_obj.set_controller(walker["con"])

                                    # 验证Walker对象是否正确存储了旋转信息
                                    if not hasattr(walker_obj, 'rotation'):
                                        # 如果Walker类没有rotation属性，手动添加
                                        walker_obj.rotation = {
                                            "pitch": carla_rotation.pitch,
                                            "yaw": carla_rotation.yaw,
                                            "roll": carla_rotation.roll
                                        }

                                    self.walkers.append(walker_obj)
                                except Exception as e:
                                    # 更详细的错误日志，帮助定位问题
                                    print(f"创建Walker实例失败: {str(e)}, 行人ID: {walker['id']}")
                    print(f"行人生成完成，共成功生成 {len(self.walkers)}/{NUM_OF_WALKERS} 个")
                else:
                    print("所有行人生成都失败了")
            else:
                print("未找到有效行人生成点，跳过行人生成")

        # 1. 根据配置创建传感器实例列表（类型、安装位置等由配置指定）
        # 所有传感器均挂载到主车（attach_to=self.ego_vehicle.get_actor()）
        self.sensors = [Sensor(world=self.world, attach_to=self.ego_vehicle.get_actor(), **sensor_config) for
                        sensor_config in scene_config["calibrated_sensors"]["sensors"]]
        sensors_batch = [SpawnActor(sensor.blueprint, sensor.transform, sensor.attach_to) for sensor in self.sensors]
        for i, response in enumerate(self.client.apply_batch_sync(sensors_batch)):
            if not response.error:
                self.sensors[i].set_actor(response.actor_id)
            else:
                print(response.error)
        self.sensors = list(filter(lambda sensor: sensor.get_actor(), self.sensors))

    # def generate_custom_scene(self,scene_config):
    #
    #     if scene_config["weather_mode"] == "custom":# 根据配置选择天气模式（自定义参数或预设模式）
    #         self.weather = carla.WeatherParameters(**scene_config["weather"])
    #     else:
    #         self.weather = getattr(carla.WeatherParameters, scene_config["weather_mode"])
    #
    #     self.world.set_weather(self.weather) # 应用天气设置
    #     SpawnActor = carla.command.SpawnActor #定义 “生成实体” 的指令类。
    #     SetAutopilot = carla.command.SetAutopilot #定义 “启用自动驾驶” 的指令类。
    #     FutureActor = carla.command.FutureActor #定义 “未来实体” 的引用类，用于关联批量指令中的依赖关系。
    #
    #     self.ego_vehicle = Vehicle(world=self.world,**scene_config["ego_vehicle"])# 1. 根据配置创建主车实例（车型、初始位置等）
    #     self.ego_vehicle.blueprint.set_attribute('role_name', 'hero')# 2. 标记为主车（Carla 中用 "hero" 标识，便于后续识别）
    #     self.ego_vehicle.spawn_actor()# 3. 在 Carla 世界中生成主车实体
    #     self.ego_vehicle.get_actor().set_autopilot()# 4. 启用主车自动驾驶
    #
    #     # 5. 配置交通管理器（让主车无视交通规则，按预设路径行驶）
    #     self.trafficmanager.ignore_lights_percentage(self.ego_vehicle.get_actor(),100) # 无视红绿灯
    #     self.trafficmanager.ignore_signs_percentage(self.ego_vehicle.get_actor(),100)# 无视交通标志
    #     self.trafficmanager.ignore_vehicles_percentage(self.ego_vehicle.get_actor(),100)  # 无视其他车辆
    #     self.trafficmanager.distance_to_leading_vehicle(self.ego_vehicle.get_actor(),0) # 与前车距离为 0（贴紧行驶）
    #     self.trafficmanager.vehicle_percentage_speed_difference(self.ego_vehicle.get_actor(),-20)  # 超速 20%（负值表示超速）
    #     self.trafficmanager.auto_lane_change(self.ego_vehicle.get_actor(), True) # 允许自动变道
    #
    #     # 1. 根据配置创建车辆实例列表（每个车辆的车型、初始位置等由配置指定）
    #     self.vehicles = [Vehicle(world=self.world,**vehicle_config) for vehicle_config in scene_config["vehicles"]]
    #     # 2. 构建批量生成指令：生成车辆 + 启用自动驾驶
    #     vehicles_batch = [SpawnActor(vehicle.blueprint,vehicle.transform)
    #                         .then(SetAutopilot(FutureActor, True, self.trafficmanager.get_port()))
    #                         for vehicle in self.vehicles]
    #
    #     # 3. 执行批量指令并处理结果
    #     for i,response in enumerate(self.client.apply_batch_sync(vehicles_batch)):
    #         if not response.error:
    #             # 生成成功：记录车辆的 Actor ID
    #             self.vehicles[i].set_actor(response.actor_id)
    #         else:
    #             # 生成失败：打印错误信息（如位置被占用）
    #             print(response.error)
    #     # 4. 过滤掉生成失败的车辆（保留有有效 Actor 的车辆）
    #     self.vehicles = list(filter(lambda vehicle:vehicle.get_actor(),self.vehicles))
    #
    #     # 5. 为每个车辆设置预设路径（由配置中的 "path" 字段指定）
    #     for vehicle in self.vehicles:
    #         self.trafficmanager.set_path(vehicle.get_actor(),vehicle.path)
    #
    #     # 1. 根据配置创建行人实例列表（初始位置、模型等由配置指定）
    #     self.walkers = [Walker(world=self.world,**walker_config) for walker_config in scene_config["walkers"]]
    #     # 2. 批量生成行人实体
    #     walkers_batch = [SpawnActor(walker.blueprint,walker.transform) for walker in self.walkers]
    #     for i,response in enumerate(self.client.apply_batch_sync(walkers_batch)):
    #         if not response.error:
    #             # 生成成功：记录行人的 Actor ID
    #             self.walkers[i].set_actor(response.actor_id)
    #         else:
    #             print(response.error)
    #     # 3. 过滤掉生成失败的行人
    #     self.walkers = list(filter(lambda walker:walker.get_actor(),self.walkers))
    #
    #     # 4. 为行人添加 AI 控制器（使行人能够行走）
    #     walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
    #     walkers_controller_batch = [SpawnActor(walker_controller_bp,carla.Transform(),walker.get_actor()) for walker in self.walkers]
    #     for i,response in enumerate(self.client.apply_batch_sync(walkers_controller_batch)):
    #                 if not response.error:
    #                     # 控制器生成成功：记录控制器 ID
    #                     self.walkers[i].set_controller(response.actor_id)
    #                 else:
    #                     print(response.error)
    #     # 5. 启动行人行走（让控制器生效）
    #     self.world.tick()
    #     for walker in self.walkers:
    #         walker.start()# 开始按预设路径行走
    #
    #     # 1. 根据配置创建传感器实例列表（类型、安装位置等由配置指定）
    #     # 所有传感器均挂载到主车（attach_to=self.ego_vehicle.get_actor()）
    #     self.sensors = [Sensor(world=self.world, attach_to=self.ego_vehicle.get_actor(), **sensor_config) for sensor_config in scene_config["calibrated_sensors"]["sensors"]]
    #     sensors_batch = [SpawnActor(sensor.blueprint,sensor.transform,sensor.attach_to) for sensor in self.sensors]
    #     for i,response in enumerate(self.client.apply_batch_sync(sensors_batch)):
    #         if not response.error:
    #             self.sensors[i].set_actor(response.actor_id)
    #         else:
    #             print(response.error)
    #     self.sensors = list(filter(lambda sensor:sensor.get_actor(),self.sensors))

    def tick(self):
        self.world.tick()

    def generate_random_scene(self,scene_config):
        print("generate random scene start!")
        self.weather = carla.WeatherParameters(**self.get_random_weather())
        self.world.set_weather(self.weather)


        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        
        
        ego_bp_name=scene_config["ego_bp_name"]
        ego_location={attr:getattr(spawn_points[0].location,attr) for attr in ["x","y","z"]}
        ego_rotation={attr:getattr(spawn_points[0].rotation,attr) for attr in ["yaw","pitch","roll"]}
        self.ego_vehicle = Vehicle(world=self.world,bp_name=ego_bp_name,location=ego_location,rotation=ego_rotation)
        self.ego_vehicle.blueprint.set_attribute('role_name', 'hero')
        self.ego_vehicle.spawn_actor()
        self.ego_vehicle.get_actor().set_autopilot()
        self.trafficmanager.ignore_lights_percentage(self.ego_vehicle.get_actor(),100)
        self.trafficmanager.ignore_signs_percentage(self.ego_vehicle.get_actor(),100)
        self.trafficmanager.ignore_vehicles_percentage(self.ego_vehicle.get_actor(),100)
        self.trafficmanager.distance_to_leading_vehicle(self.ego_vehicle.get_actor(),0)
        self.trafficmanager.vehicle_percentage_speed_difference(self.ego_vehicle.get_actor(),-20)
        self.trafficmanager.auto_lane_change(self.ego_vehicle.get_actor(), True)

        vehicle_bp_list = self.world.get_blueprint_library().filter("vehicle")
        self.vehicles = []
        for spawn_point in spawn_points[1:random.randint(1,len(spawn_points))]:
            location = {attr:getattr(spawn_point.location,attr) for attr in ["x","y","z"]}
            rotation = {attr:getattr(spawn_point.rotation,attr) for attr in ["yaw","pitch","roll"]}
            bp_name = random.choice(vehicle_bp_list).id
            self.vehicles.append(Vehicle(world=self.world,bp_name=bp_name,location=location,rotation=rotation))
        vehicles_batch = [SpawnActor(vehicle.blueprint,vehicle.transform)
                            .then(SetAutopilot(FutureActor, True, self.trafficmanager.get_port())) 
                            for vehicle in self.vehicles]

        for i,response in enumerate(self.client.apply_batch_sync(vehicles_batch)):
            if not response.error:
                self.vehicles[i].set_actor(response.actor_id)
            else:
                print(response.error)
        self.vehicles = list(filter(lambda vehicle:vehicle.get_actor(),self.vehicles))

        walker_bp_list = self.world.get_blueprint_library().filter("pedestrian")
        self.walkers = []
        for i in range(random.randint(len(spawn_points),len(spawn_points)*2)):
            spawn = self.world.get_random_location_from_navigation()
            if spawn != None:
                bp_name=random.choice(walker_bp_list).id
                spawn_location = {attr:getattr(spawn,attr) for attr in ["x","y","z"]}
                destination=self.world.get_random_location_from_navigation()
                destination_location={attr:getattr(destination,attr) for attr in ["x","y","z"]}
                rotation = {"yaw":random.random()*360,"pitch":random.random()*360,"roll":random.random()*360}
                self.walkers.append(Walker(world=self.world,location=spawn_location,rotation=rotation,destination=destination_location,bp_name=bp_name))
            else:
                print("walker generate fail")
        walkers_batch = [SpawnActor(walker.blueprint,walker.transform) for walker in self.walkers]
        for i,response in enumerate(self.client.apply_batch_sync(walkers_batch)):
            if not response.error:
                self.walkers[i].set_actor(response.actor_id)
            else:
                print(response.error)
        self.walkers = list(filter(lambda walker:walker.get_actor(),self.walkers))

        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        walkers_controller_batch = [SpawnActor(walker_controller_bp,carla.Transform(),walker.get_actor()) for walker in self.walkers]
        for i,response in enumerate(self.client.apply_batch_sync(walkers_controller_batch)):
                    if not response.error:
                        self.walkers[i].set_controller(response.actor_id)
                    else:
                        print(response.error)
        self.world.tick()
        for walker in self.walkers:
            walker.start()

        self.sensors = [Sensor(world=self.world, attach_to=self.ego_vehicle.get_actor(), **sensor_config) for sensor_config in scene_config["calibrated_sensors"]["sensors"]]
        sensors_batch = [SpawnActor(sensor.blueprint,sensor.transform,sensor.attach_to) for sensor in self.sensors]
        for i,response in enumerate(self.client.apply_batch_sync(sensors_batch)):
            if not response.error:
                self.sensors[i].set_actor(response.actor_id)
            else:
                print(response.error)
        self.sensors = list(filter(lambda sensor:sensor.get_actor(),self.sensors))
        print("generate random scene success!")        

    def destroy_scene(self):
        if self.walkers is not None:
            for walker in self.walkers:
                walker.controller.stop()
                walker.destroy()
        if self.vehicles is not None:
            for vehicle in self.vehicles:
                vehicle.destroy()
        if self.sensors is not None:
            for sensor in self.sensors:
                sensor.destroy()
        if self.ego_vehicle is not None:
            self.ego_vehicle.destroy()


    def destroy_world(self):
        self.trafficmanager.set_synchronous_mode(False)
        self.ego_vehicle = None
        self.sensors = None
        self.vehicles = None
        self.walkers = None
        self.world.apply_settings(self.original_settings)

    def get_calibrated_sensor(self,sensor):
        sensor_token = generate_token("sensor",sensor.name)
        channel = sensor.name
        if sensor.bp_name == "sensor.camera.rgb":
            intrinsic = get_intrinsic(float(sensor.get_actor().attributes["fov"]),
                            float(sensor.get_actor().attributes["image_size_x"]),
                            float(sensor.get_actor().attributes["image_size_y"])).tolist()
            rotation,translation = get_nuscenes_rt(sensor.transform,"zxy")
        else:
            intrinsic = []
            rotation,translation = get_nuscenes_rt(sensor.transform)
        return sensor_token,channel,translation,rotation,intrinsic
        
    def get_ego_pose(self,sample_data):
        timestamp = transform_timestamp(sample_data[1].timestamp)
        rotation,translation = get_nuscenes_rt(sample_data[0])
        return timestamp,translation,rotation
    
    def get_sample_data(self,sample_data):
        height = 0
        width = 0
        if isinstance(sample_data[1],carla.Image):
            height = sample_data[1].height
            width = sample_data[1].width
        return sample_data,height,width

    def get_sample(self):
        return (transform_timestamp(self.world.get_snapshot().timestamp.elapsed_seconds),)

    def get_instance(self,scene_token,instance):
        category_token = generate_token("category",self.category_dict[instance.blueprint.id])
        id = hash((scene_token,instance.get_actor().id))
        return category_token,id

    def get_sample_annotation(self,scene_token,instance):
        instance_token = generate_token("instance",hash((scene_token,instance.get_actor().id)))
        visibility_token = str(self.get_visibility(instance))
        
        attribute_tokens = [generate_token("attribute",attribute) for attribute in self.get_attributes(instance)]
        rotation,translation = get_nuscenes_rt(instance.get_transform())
        size = [instance.get_size().y,instance.get_size().x,instance.get_size().z]#xyz to whl
        num_lidar_pts = 0
        num_radar_pts = 0
        for sensor in self.sensors:
            if sensor.bp_name == 'sensor.lidar.ray_cast':
                num_lidar_pts += self.get_num_lidar_pts(instance,sensor.get_last_data(),sensor.get_transform())
            elif sensor.bp_name == 'sensor.other.radar':
                num_radar_pts += self.get_num_radar_pts(instance,sensor.get_last_data(),sensor.get_transform())
        return instance_token,visibility_token,attribute_tokens,translation,rotation,size,num_lidar_pts,num_radar_pts

    def get_visibility(self,instance):
        max_visible_point_count = 0
        for sensor in self.sensors:
            if sensor.bp_name == 'sensor.lidar.ray_cast':
                ego_position = sensor.get_transform().location
                ego_position.z += self.ego_vehicle.get_size().z*0.5
                instance_position = instance.get_transform().location
                visible_point_count1 = 0
                visible_point_count2 = 0
                for i in range(5):
                    size = instance.get_size()
                    size.z = 0
                    check_point = instance_position-(i-2)*size*0.5
                    ray_points =  self.world.cast_ray(ego_position,check_point)
                    points = list(filter(lambda point:not self.ego_vehicle.get_actor().bounding_box.contains(point.location,self.ego_vehicle.get_actor().get_transform()) 
                                        and not instance.get_actor().bounding_box.contains(point.location,instance.get_actor().get_transform()) 
                                        and point.label is not carla.libcarla.CityObjectLabel.NONE,ray_points))
                    if not points:
                        visible_point_count1+=1
                    size.x = -size.x
                    check_point = instance_position-(i-2)*size*0.5
                    ray_points =  self.world.cast_ray(ego_position,check_point)
                    points = list(filter(lambda point:not self.ego_vehicle.get_actor().bounding_box.contains(point.location,self.ego_vehicle.get_actor().get_transform()) 
                                        and not instance.get_actor().bounding_box.contains(point.location,instance.get_actor().get_transform()) 
                                        and point.label is not carla.libcarla.CityObjectLabel.NONE,ray_points))
                    if not points:
                        visible_point_count2+=1
                if max(visible_point_count1,visible_point_count2)>max_visible_point_count:
                    max_visible_point_count = max(visible_point_count1,visible_point_count2)
        visibility_dict = {0:0,1:1,2:1,3:2,4:3,5:4}
        return visibility_dict[max_visible_point_count]

    def get_attributes(self,instance):
        return self.attribute_dict[instance.bp_name]

    def get_num_lidar_pts(self,instance,lidar_data,lidar_transform):
        num_lidar_pts = 0
        if lidar_data is not None:
            for data in lidar_data[1]:
                point = lidar_transform.transform(data.point)
                if instance.get_actor().bounding_box.contains(point,instance.get_actor().get_transform()):
                    num_lidar_pts+=1
        return num_lidar_pts

    def get_num_radar_pts(self,instance,radar_data,radar_transform):
        num_radar_pts = 0
        if radar_data is not None:
            for data in radar_data[1]:
                point = carla.Location(data.depth*math.cos(data.altitude)*math.cos(data.azimuth),
                        data.depth*math.sin(data.altitude)*math.cos(data.azimuth),
                        data.depth*math.sin(data.azimuth)
                        )
                point = radar_transform.transform(point)
                if instance.get_actor().bounding_box.contains(point,instance.get_actor().get_transform()):
                    num_radar_pts+=1
        return num_radar_pts

    def get_random_weather(self):
        weather_param = {
            "cloudiness":clamp(random.gauss(0,30)),
            "sun_azimuth_angle":random.random()*360,
            "sun_altitude_angle":random.random()*120-30,
            "precipitation":clamp(random.gauss(0,30)),
            "precipitation_deposits":clamp(random.gauss(0,30)),
            "wind_intensity":random.random()*100,
            "fog_density":clamp(random.gauss(0,30)),
            "fog_distance":random.random()*100,
            "wetness":clamp(random.gauss(0,30)),
            "fog_falloff":random.random()*5,
            "scattering_intensity":max(random.random()*2-1,0),
            "mie_scattering_scale":max(random.random()*2-1,0),
            "rayleigh_scattering_scale":max(random.random()*2-1,0),
            "dust_storm":clamp(random.gauss(0,30))
        }
        return weather_param

    