import numpy as np
import random
import os
import warnings
from dataset import load_scenario_npy, listdir_limited, load_scenario
import math
from datetime import datetime, timedelta


def two_partitions_with_negatives(max_n):
    """
    Generate all possible combinations of two partitions of n, with negatives allowed, and yield them in a random order.
    """
    n = 0
    while n <= max_n:
        for a_abs in range(n, -1, -1):
            b_abs = n - a_abs
            a_candidates = [a_abs] if a_abs == 0 else [a_abs, -a_abs]
            b_candidates = [b_abs] if b_abs == 0 else [b_abs, -b_abs]
            # shuffle the candidates
            candidates = [(a, b) for a in a_candidates for b in b_candidates]
            random.shuffle(candidates)
            for a, b in candidates:
                yield (a, b)
        n += 1


class ScenarioSampler:
    def __init__(self, scenario_folder, extension = '.npy'):
        """
        Initialize the sampler: build ignition point -> scenarios mapping.
        
        Args:
            scenario_folder (str): path to the folder containing scenarios
        """
        self.scenario_folder = scenario_folder
        self.ignition_map = {}  # Maps (x,y) -> list of scenario filenames
        self._build_index(extension = extension)
        
    def _build_index(self, extension = '.npy'):
        """
        Build the ignition_map from the scenarios.
        """
        for filename in listdir_limited(self.scenario_folder):
            if extension == '.npy' and not filename.endswith('.npy') or filename == '.DS_Store':
                continue
            try:
                scenario_frame = load_scenario(os.path.join(self.scenario_folder, filename), extension = extension, first_frame_only=True)
                fire_points = np.argwhere(scenario_frame)  # first frame
                if fire_points.shape[0] == 0:
                    continue
                for x, y in fire_points:  # <==== iterate over ALL fire points
                    key = (x, y)
                    if key not in self.ignition_map:
                        self.ignition_map[key] = []
                    self.ignition_map[key].append(filename)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        print(f"Sampler built with {len(self.ignition_map)} ignition points.")

    def get_scenario_location(self, ignition_point, leeway_distance, sampling_method='random', exclude_scenarios=[]):
        """
        Sample a scenario starting near a given point.
        
        Args:
            ignition_point (tuple): (x,y) coordinates
            leeway_distance (int): maximum allowed l_inf distance
            sampling_method (str): 'random' or 'closest'
        Returns:
            str: Filename of the selected scenario
        """
        candidates = []
        closest_scenario = None
        closest_distance = float('inf')
        x0, y0 = ignition_point
        closest_ignition_point = None
        for (x, y), filenames in self.ignition_map.items():
            if max(abs(x - x0), abs(y - y0)) <= leeway_distance:
                new_candidates = [filename for filename in filenames if filename.split('.')[0] not in exclude_scenarios]
                candidates.extend(new_candidates)
                if sampling_method == 'closest' and len(new_candidates) > 0:
                    distance = np.linalg.norm(np.array([x, y]) - np.array([x0, y0]))
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_scenario = new_candidates[0]
                        closest_ignition_point = (x, y)
        if not candidates:
            # warnings.warn(f"No scenario found around {ignition_point} with leeway {leeway_distance}")
            return None, None
        
        if sampling_method == 'random':
            return random.choice(candidates)
        elif sampling_method == 'closest':
            return closest_scenario, closest_ignition_point



class ScenarioSamplerDate:
    def __init__(self, scenario_folder, extension = '.npy'):
        """
        Initialize the sampler: build ignition point -> scenarios mapping.
        
        Args:
            scenario_folder (str): path to the folder containing scenarios
        """
        self.scenario_folder = scenario_folder
        self.ignition_map = {}  # Maps date -> [{ignition_points}, filename)]
        self._build_index(extension = extension)

    def _build_index(self, extension = '.npy'):
        """
        Build the ignition_map from the scenarios.
        """
        for filename in listdir_limited(self.scenario_folder):
            if extension == '.npy' and not filename.endswith('.npy') or filename == '.DS_Store':
                continue
            try:
                scenario_frame = load_scenario(os.path.join(self.scenario_folder, filename), extension = extension, first_frame_only=True)
                
                # get the date of the scenario from weather file
                parent_folder = '/'.join(self.scenario_folder.split('/')[:-2])
                weather_file = os.path.join(parent_folder, 'Weather_Data', filename + '.txt')

                if not os.path.exists(weather_file):
                    continue
                with open(weather_file, 'r') as f:
                    line1 = f.readline()
                    date = line1[:10]
                    date = datetime.strptime(date, '%Y %m %d')

                if date not in self.ignition_map:
                    self.ignition_map[date] = []

                fire_points = set(tuple(p) for p in np.argwhere(scenario_frame))

                if len(fire_points) == 0:
                    continue
                self.ignition_map[date].append((fire_points, filename))

            except Exception as e:
                print(f"Error processing {filename}: {e}")
        print(f"Sampler built with {len(self.ignition_map)} ignition dates.")
        #print(self.ignition_map[datetime(2023, 8, 4)])

    def get_scenario_location(self, ignition_point, date, leeway_distance, leeway_date, sampling_method='clostest', exclude_scenarios=[]):
        """
        Sample a scenario starting near a given point.
        
        Args:
            ignition_point (tuple): (x,y) coordinates
            leeway_distance (int): maximum allowed l_inf distance
            sampling_method (str): 'random' or 'closest'
        Returns:
            str: Filename of the selected scenario
        """
        candidates = []
        d0 = date
        x0, y0 = ignition_point
        shift_generator = two_partitions_with_negatives(leeway_distance)
        dateshift = 0

        found = False
        # priority is same date, then closest distance
        while not found:
            try:
                x_shift, y_shift = next(shift_generator)
            except StopIteration:
                dateshift += 1
                if dateshift > leeway_date:
                    break
                date = d0 + timedelta(days=math.ceil(dateshift / 2) * (1 if dateshift % 2 == 0 else -1))
                shift_generator = two_partitions_with_negatives(leeway_distance)
            x,y = x0 + x_shift, y0 + y_shift
            print(x,y)
            for (fire_points, filename) in self.ignition_map.get(date, []):
                print(filename)
                if (x,y) in fire_points:
                    if filename.split('.')[0] not in exclude_scenarios:
                        candidates.append(filename)
                        distance = abs(x_shift) + abs(y_shift)
                        ignition_point = (x,y)
                        ignition_date = date
                        found = True
                    break
            
        #print(self.ignition_map.keys())

        if not candidates:
            # warnings.warn(f"No scenario found around {ignition_point} with leeway {leeway_distance}")
            return None, None
        
        if sampling_method == 'random':
            return random.choice(candidates)
        elif sampling_method == 'closest':
            return candidates[0], ignition_point, ignition_date


if __name__ == "__main__":
    import numpy as np
    import os


    test_folder = "./WideDataset/0002_00714/Satellite_Images_Mask/"
    #os.makedirs(test_folder, exist_ok=True)

    # === Step 2: Initialize the sampler ===
    sampler = ScenarioSamplerDate(test_folder, extension = '.jpg')
    # print(f"Sampler built with {len(sampler.ignition_map)} ignition points.")
    # print(f"Sampled ignition points: {sampler.ignition_map}")
    # === Step 3: Test getting a scenario ===
    try:
        ignition_point = (96, 161)
        ignition_date = datetime(2023, 8, 4)
        leeway = 2000
        selected_scenario = sampler.get_scenario_location(ignition_point, ignition_date, leeway, 0, sampling_method='closest') #exclude_scenarios=['0002_00100', '0002_00026'])
        print(f"Selected scenario near {ignition_point} (leeway {leeway}): {selected_scenario}")
    except ValueError as e:
        print(f"Error: {e}")