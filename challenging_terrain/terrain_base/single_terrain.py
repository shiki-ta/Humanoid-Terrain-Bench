import numpy as np
from numpy.random import choice
from scipy import interpolate
import random
from isaacgym import terrain_utils
from .config import terrain_config
from scipy import ndimage
from pydelatin import Delatin
import pyfqmr
from scipy.ndimage import binary_dilation
import math

class single_terrain:
    def __init__(self, cfg: terrain_config) -> None:
        self.cfg = cfg
    
    def parkour(terrain, 
            length_x=18.,
            length_y=4.,
            num_goals=6, 
            start_x=0,
            start_y=0,
            platform_size=2.5, 
            difficulty=0.5,
            x_range=[0.5, 1.0],
            y_range=[0.3, 0.4],
            stone_len_range=[0.8, 1.0],
            stone_width_range=[0.6, 0.8],
            incline_height=0.1,
            pit_depth=[0.5, 1.]):
    
        goals = np.zeros((num_goals, 2))
        pit_depth_val = np.random.uniform(pit_depth[0], pit_depth[1])
        pit_depth_grid = -round(pit_depth_val / terrain.vertical_scale)
        
        h_scale = terrain.horizontal_scale
        v_scale = terrain.vertical_scale
    
        length_y_grid = round(length_y / h_scale)
        mid_y = length_y_grid // 2

        length_x_grid = round(length_x / h_scale)
        
        stone_len = round(((stone_len_range[0] - stone_len_range[1]) * difficulty + stone_len_range[1]) / h_scale)
        stone_width = round(((stone_width_range[0] - stone_width_range[1]) * difficulty + stone_width_range[1]) / h_scale)
        gap_x = round(((x_range[1] - x_range[0]) * difficulty + x_range[0]) / h_scale)
        gap_y = round(((y_range[1] - y_range[0]) * difficulty + y_range[0]) / h_scale)
        
        platform_size_grid = int(round(platform_size / h_scale))
        incline_height_grid = int(round(incline_height / v_scale))
        
        terrain.height_field_raw[start_x+platform_size_grid:start_x + length_x_grid, start_y:start_y+length_y_grid*2] = pit_depth_grid
        
        dis_x = start_x +platform_size_grid - gap_x + stone_len // 2
        goals[0] = [start_x + platform_size_grid - stone_len // 2, start_y + mid_y]
        left_right_flag = np.random.randint(0, 2)
        
        for i in range(num_goals - 2):
            dis_x += gap_x
            pos_neg = 2 * (left_right_flag - 0.5)
            dis_y = mid_y + pos_neg * gap_y
            
            x_start = int(dis_x - stone_len // 2)
            x_end = x_start + stone_len
            y_start = int(dis_y - stone_width // 2)
            y_end = y_start + stone_width
            
            heights = np.tile(np.linspace(-incline_height_grid, incline_height_grid, stone_width),(stone_len, 1)) * pos_neg
            heights = heights.astype(int)
            
            if x_end > terrain.height_field_raw.shape[0]:
                x_end = terrain.height_field_raw.shape[0]
            if y_end > terrain.height_field_raw.shape[1]:
                y_end = terrain.height_field_raw.shape[1]
    
            actual_height = heights[:x_end - x_start, :y_end - y_start]
            terrain.height_field_raw[x_start:x_end, y_start:y_end] = actual_height
            goals[i + 1] = [dis_x, dis_y]
            left_right_flag = 1 - left_right_flag
        
        final_dis_x = dis_x + gap_x
        goals[-1] = [final_dis_x, mid_y]
        return terrain, goals, final_dis_x
    
    def hurdle(
            terrain,
            length_x=18.,
            length_y=4.,
            num_goals=8,
            start_x=0,
            start_y=0,
            platform_size=1., 
            difficulty = 0.5,
            hurdle_range=[0.1, 0.2],
            hurdle_height_range=[0.05, 0.15],
            flat_size = 0.6
            ):
        
        goals = np.zeros((num_goals, 2))
        mid_y = round(length_y/ terrain.horizontal_scale)// 2  
        per_x = (round(length_x/ terrain.horizontal_scale)- platform_size) // num_goals
        hurdle_size = round(((hurdle_range[1]-hurdle_range[0])*difficulty +hurdle_range[0])/terrain.horizontal_scale)
        hurdle_height = round(((hurdle_height_range[1]-hurdle_height_range[0])*difficulty + hurdle_height_range[0])/terrain.vertical_scale)

        platform_size = round(platform_size / terrain.horizontal_scale)
        terrain.height_field_raw[start_x:start_x +round(length_x/ terrain.horizontal_scale), start_y:start_y+mid_y*2] = 0

        flat_size = round(flat_size / terrain.horizontal_scale)
        dis_x = start_x + platform_size

        for i in range(num_goals):
            goals[i]=[dis_x+per_x*i,start_y+mid_y]

        for i in range(num_goals):

            terrain.height_field_raw[dis_x-hurdle_size//2:dis_x+hurdle_size//2, start_y:start_y+mid_y*2] = hurdle_height
            dis_x += flat_size + hurdle_size

        return terrain,goals,dis_x
        
    def bridge(terrain,
               length_x=18.0,
                length_y=4.0,
                num_goals=8,
                start_x = 0,
                start_y = 0,
                platform_size=1.0, 
                difficulty = 0.5,
                bridge_width_range=[0.3,0.4],  
                bridge_height=0.7,
                ):
        goals = np.zeros((num_goals, 2))
        mid_y = round(length_y / terrain.horizontal_scale) // 2  
        bridge_width = round(((bridge_width_range[1]-bridge_width_range[0])*difficulty +bridge_width_range[0])/terrain.horizontal_scale)
        bridge_height = round(bridge_height / terrain.vertical_scale)
        platform_size = round(platform_size / terrain.horizontal_scale)
        terrain.height_field_raw[start_x:start_x+platform_size, start_y:start_y+2*mid_y] = 0
        bridge_start_x = platform_size + start_x
        bridge_length = round(length_x / terrain.horizontal_scale)
        bridge_end_x = start_x + bridge_length

        for i in range(num_goals):
            goals[i] = [bridge_start_x + bridge_length/num_goals*i, mid_y]  
       
        left_y1 = 0
        left_y2 = int(mid_y - bridge_width // 2) 
        right_y1 = int(mid_y + bridge_width // 2)
        right_y2 = mid_y*2
        terrain.height_field_raw[bridge_start_x:bridge_end_x, left_y1:left_y2] = -bridge_height
        terrain.height_field_raw[bridge_start_x:bridge_end_x, right_y1:right_y2] = -bridge_height

        return terrain,goals,bridge_end_x

    def flat(terrain,
            length_x=18.0,
            length_y=4.0,
            num_goals=8,
            start_x = 0,
            start_y = 0,
            platform_size=1.0, 
            difficulty = 0.5,
            ):
        goals = np.zeros((num_goals, 2))
        length_x = round(length_x / terrain.horizontal_scale)
        length_y = round(length_y / terrain.horizontal_scale)
        platform_size = round(platform_size / terrain.horizontal_scale)

        for i in range(num_goals):
            y_pos = length_y//2
            goals[i]=[start_x+platform_size+length_x/num_goals*i,start_y+y_pos]

        return terrain,goals,length_x

    def uneven(terrain,
            length_x=18.0,
            length_y=4.0,
            num_goals=8,
            start_x = 0,
            start_y = 0,
            platform_size=1.0, 
            difficulty = 0.5,
            num_range=[150,200],
            size_range=[0.4,0.7],
            height_range=[0.1,0.2],
            ):   

        goals = np.zeros((num_goals, 2))
        platform_size = round(platform_size/ terrain.horizontal_scale)
        per_x = (round(length_x/ terrain.horizontal_scale) - platform_size)// num_goals
        mid_y = round(length_y/ terrain.horizontal_scale) // 2

        for i in range(num_goals):
            goals[i]=[start_x+platform_size+per_x*i,start_y+mid_y]

        height = round(((height_range[1]-height_range[0])*difficulty + height_range[0])/terrain.vertical_scale)


        min_size = round(size_range[0]/ terrain.horizontal_scale)
        max_size = round(size_range[1]/ terrain.horizontal_scale)

        discrete_start_x = start_x+platform_size
        discrete_start_y = start_y

        discrete_end_x = discrete_start_x +round(length_x/ terrain.horizontal_scale) - platform_size
        discrete_end_y = discrete_start_y +round(length_y/ terrain.horizontal_scale)

        num_rects = round((num_range[1]-num_range[0])*difficulty + num_range[0])

        for _ in range(num_rects):
            width = round(random.uniform(min_size, max_size))
            length = round(random.uniform(min_size, max_size))
            start_i = round(random.uniform(discrete_start_x, discrete_end_x-width))
            start_j = round(random.uniform(discrete_start_y, discrete_end_y-length))

            terrain.height_field_raw[start_i:start_i+width, start_j:start_j+length] = random.uniform(-height//2, height)

        terrain.height_field_raw[start_x:start_x+platform_size , start_y:start_y+mid_y*2] = 0
        terrain.height_field_raw[discrete_end_x:discrete_end_x+platform_size , start_y:start_y+mid_y*2] = 0

        return terrain,goals,discrete_end_x+platform_size

    def stair(terrain,
                length_x=18.0,
                length_y=4.0,
                num_goals=8,
                start_x = 0,
                start_y = 0,
                platform_size=1.0, 
                difficulty = 0.5,
                height_range=[0.08,0.2],
                size_range=[0.4,0.5],
                upstair = True,
                start_z = 3.0
                ):

        goals = np.zeros((num_goals, 2))
        platform_size = round(platform_size/ terrain.horizontal_scale)
        per_x = (round(length_x/ terrain.horizontal_scale)- platform_size) // num_goals
        per_y = round(length_y/ terrain.horizontal_scale) // 2
        step_height = round(((height_range[1]-height_range[0])*difficulty + height_range[0])/terrain.vertical_scale)
        step_x = round(((size_range[0]-size_range[1])*difficulty +size_range[1])/terrain.horizontal_scale)

        if(upstair):
            total_step_height = 0
        else:
            total_step_height = round(start_z/terrain.vertical_scale)

        dis_x = start_x + platform_size

        for i in range(num_goals):
            goals[i]=[dis_x+per_x*i,start_y+per_y]

        for i in range(num_goals):
            if(upstair):
                total_step_height += step_height
            else :
                total_step_height -= step_height

            terrain.height_field_raw[dis_x : dis_x + step_x, start_y : start_y + per_y*2] = total_step_height
            dis_x += step_x
        terrain.height_field_raw[dis_x:start_x+round(length_x/ terrain.horizontal_scale),start_y:start_y + per_y*2] = total_step_height
        return terrain,goals,start_x+round(length_x/ terrain.horizontal_scale)

    def wave(terrain,
            length_x=18.0,
            length_y=4.0,
            num_goals=8,
            start_x = 0,
            start_y = 0,
            platform_size=1.0, 
            difficulty = 0.5,
            amplitude_range=[0.05,0.1]
            ):   
        goals = np.zeros((num_goals, 2))
        mid_y = round(length_y/ terrain.horizontal_scale) //2
        platform_size = round(1.5/ terrain.horizontal_scale)
        mid_x =  (round(length_x/ terrain.horizontal_scale) - platform_size)// num_goals

        for i in range(num_goals):
            goals[i]=[start_x+platform_size+mid_x*i,start_y+mid_y]
        
        x_indices = np.arange(start_x, start_x + mid_x*num_goals + platform_size)
        amplitude = round(((amplitude_range[1]-amplitude_range[0])*difficulty + amplitude_range[0])/terrain.vertical_scale)
        wave_pattern = amplitude * np.sin(2 * np.pi * x_indices / length_x)

        for i, wave_height in enumerate(wave_pattern):
            terrain.height_field_raw[x_indices[i], start_y:start_y +mid_y*2] = wave_height

        terrain.height_field_raw[start_x :start_x + platform_size, start_y:start_y+ mid_y*2] = 0

        return terrain,goals,start_x+mid_x*num_goals

    def slope(terrain,
            length_x=18.0,
            length_y=4.0,
            num_goals=8,
            start_x = 0,
            start_y = 0,
            platform_size=1.0, 
            difficulty = 0.5,
            angle_range = [4.1,10.0],
            uphill=False
            ):    

        goals = np.zeros((num_goals, 2))
        length_x_grid = round((length_x - platform_size) / terrain.horizontal_scale)
        length_y_grid = round(length_y / terrain.horizontal_scale)
        platform_size = round(platform_size/ terrain.horizontal_scale)

        for i in range(num_goals):
            goals[i]=[start_x+platform_size+length_x_grid/num_goals*i,start_y+length_y_grid//2]

        slope_angle = (angle_range[1]-angle_range[0])*difficulty + angle_range[0]
        angle_rad = math.radians(slope_angle)
        total_height = length_x * math.tan(angle_rad)

        total_height_units = total_height / terrain.vertical_scale

        start_x += platform_size

        for x in range(start_x, start_x + length_x_grid):
            progress = (x - start_x) / length_x_grid
            if uphill:
                height = progress * total_height_units
            else:
                height = (1 - progress) * total_height_units
            terrain.height_field_raw[x, start_y:start_y + length_y_grid] = round(height)
        
        return terrain,goals,start_x + length_x_grid

    def gap(terrain,
            length_x=18.0,
            length_y=4.0,
            num_goals=8,
            start_x = 0,
            start_y = 0,
            platform_size=1.0,
            difficulty = 0.5,
            gap_height = 2.,
            gap_low_range = [0.15,0.3],
            ):
        
        goals = np.zeros((num_goals, 2))
        mid_y = round(length_y/ terrain.horizontal_scale) //2
        mid_x =  round((length_x - platform_size)/ terrain.horizontal_scale) // num_goals
        platform_size = round(platform_size/ terrain.horizontal_scale)

        for i in range(num_goals):
            goals[i]=[start_x+platform_size+mid_x*i,start_y+mid_y]

        gap_size = round(( (gap_low_range[0]-gap_low_range[1])*difficulty + gap_low_range[1] )/terrain.horizontal_scale)
        gap_dis_x = start_x + platform_size + gap_size
        gap_dis_y = start_y + mid_y
        
        for i in range(num_goals):
            terrain.height_field_raw[gap_dis_x :gap_dis_x + gap_size, gap_dis_y - mid_y:gap_dis_y + mid_y] = -round(gap_height / terrain.vertical_scale)
            gap_dis_x += 3*gap_size
        
        terrain.height_field_raw[start_x :start_x + platform_size, start_y :start_y + mid_y*2] = 0

        return terrain, goals,start_x+mid_x*num_goals
    
    def plot(
            terrain,
            length_x=18.,
            length_y=4.,
            num_goals=8,
            start_x=0,
            start_y=0,
            platform_size=1., 
            difficulty = 0.5,
            hurdle_range=[0.1, 0.15],
            hurdle_height = 1.2,
            flat_size = 1.0
            ):
        
        goals = np.zeros((num_goals, 2))
        mid_y = round(length_y/ terrain.horizontal_scale)// 2  
        per_x = (round(length_x/ terrain.horizontal_scale)- platform_size) // num_goals
        hurdle_size = round(((hurdle_range[1]-hurdle_range[0])*difficulty +hurdle_range[0])/terrain.horizontal_scale)// 2
        hurdle_height = round(hurdle_height/terrain.vertical_scale)
        platform_size = round(platform_size / terrain.horizontal_scale)

        terrain.height_field_raw[start_x:start_x +round(length_x/ terrain.horizontal_scale), start_y:start_y+mid_y*2] = 0

        flat_size = round(flat_size / terrain.horizontal_scale)
        dis_x = start_x + platform_size

        for i in range(num_goals):
            goals[i]=[dis_x+per_x*i,start_y+mid_y]

        for i in range(num_goals):

            terrain.height_field_raw[dis_x-hurdle_size:dis_x+hurdle_size, start_y+mid_y - hurdle_size:start_y+mid_y + hurdle_size] = hurdle_height
            dis_x += flat_size + hurdle_size * 2

        return terrain,goals,dis_x
