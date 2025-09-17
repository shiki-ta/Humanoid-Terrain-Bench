from .single_terrain import single_terrain
from .config import terrain_config
import numpy as np
class combine_config:
        single = [
                single_terrain.parkour, #0
                single_terrain.hurdle,#1
                single_terrain.bridge,#2
                single_terrain.flat,#3
                single_terrain.uneven,#4
                single_terrain.stair,#5
                single_terrain.wave,#6
                single_terrain.slope,#7
                single_terrain.gap,#8
                single_terrain.plot#9
        ]

        multiplication = [
                [single[8],single[2],],#gap_bridge,0
                [single[6],single[8]],#wave_gap,1
                [single[6],single[2]],#wave_bridge,2
                [single[8],single[6],single[5],single[2]],#wave_bridge,3
                [single[5],single[6],single[2]],#wave_bridge,3
        ]

        addition = [
                [single[5],single[2],single[4],single[8]]
        ]

        proportions = [
                ("single", 3, 0.3),#
                # ("addition", 0, 0.5),
                # ("multiplication", 3, 0.4),
        ]

class generator:
        def __init__(self, cfg: terrain_config) -> None:
                self.cfg = cfg

        def single_create(terrain,id=0,difficulty=0.5):
                length_x = terrain_config.terrain_length
                length_y = terrain_config.terrain_width
                num_goals = terrain_config.num_goals
                horizontal_scale = terrain.horizontal_scale
                platform_size = terrain_config.platform_size
                terrain , goals, final_x= combine_config.single[id](
                                                                terrain=terrain, 
                                                                length_x=length_x, 
                                                                length_y=length_y, 
                                                                num_goals=num_goals, 
                                                                platform_size=platform_size, 
                                                                difficulty=difficulty)
                terrain.goals = goals * horizontal_scale
                terrain.idx = id
                return terrain

        def addition_create(terrain,id=0,difficulty=0.5):
                terrain_list = combine_config.addition[id]
                num_terrain = len(terrain_list)
                platform_size = terrain_config.platform_size
                length_x = (terrain_config.terrain_length) // num_terrain
                length_y = terrain_config.terrain_width
                num_goals = terrain_config.num_goals // num_terrain
                horizontal_scale = terrain.horizontal_scale
                goals = []
                final_x = 20
                for i in range(num_terrain):
                        if(i == num_terrain-1):
                                num_goals = terrain_config.num_goals - i*num_goals
                        terrain , sub_goals, final_x = terrain_list[i](
                                                                        terrain=terrain, 
                                                                        length_x=length_x, 
                                                                        length_y=length_y, 
                                                                        num_goals=num_goals, 
                                                                        start_x=final_x, 
                                                                        platform_size=platform_size, 
                                                                        difficulty=difficulty)
                        final_x -= round(platform_size / horizontal_scale)
                        goals.append(sub_goals)

                goals = np.vstack(goals)
                terrain.goals = goals * horizontal_scale
                terrain.idx = id + len(combine_config.single)
                return terrain

        def multiplication_create(terrain,id=0,difficulty=0.5):

                terrain_list = combine_config.multiplication[id]
                num_terrain = len(terrain_list)
                platform_size = terrain_config.platform_size
                length_x = terrain_config.terrain_length- platform_size
                length_y = terrain_config.terrain_width
                num_goals = terrain_config.num_goals
                horizontal_scale = terrain.horizontal_scale
                goals = []
                start_x = 20
                for i in range(num_terrain):
                        terrain , goals, final_x = terrain_list[i](
                                                                terrain=terrain, 
                                                                length_x=length_x, 
                                                                length_y=length_y, 
                                                                num_goals=num_goals, 
                                                                start_x=start_x, 
                                                                platform_size=platform_size, 
                                                                difficulty=difficulty)

                terrain.goals = goals * horizontal_scale
                terrain.idx = id+ len(combine_config.single) + len(combine_config.addition)
                return terrain
