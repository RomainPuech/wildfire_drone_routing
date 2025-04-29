import numpy as np
import random
import os
from dataset import load_scenario_npy, listdir_limited

class ScenarioSampler:
    def __init__(self, scenario_folder):
        """
        Initialize the sampler: build ignition point -> scenarios mapping.
        
        Args:
            scenario_folder (str): path to the folder containing scenarios
        """
        self.scenario_folder = scenario_folder
        self.ignition_map = {}  # Maps (x,y) -> list of scenario filenames
        self._build_index()
        
    def _build_index(self):
        """
        Build the ignition_map from the scenarios.
        """
        for filename in listdir_limited(self.scenario_folder):
            if not filename.endswith('.npy'):
                continue
            try:
                scenario = load_scenario_npy(os.path.join(self.scenario_folder, filename))
                fire_points = np.argwhere(scenario[0])  # first frame
                if fire_points.shape[0] == 0:
                    continue
                for x, y in fire_points:  # <==== iterate over ALL fire points
                    key = (x, y)
                    if key not in self.ignition_map:
                        self.ignition_map[key] = []
                    self.ignition_map[key].append(filename)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    def get_scenario_location(self, ignition_point, leeway_distance):
        """
        Sample a scenario starting near a given point.
        
        Args:
            ignition_point (tuple): (x,y) coordinates
            leeway_distance (int): maximum allowed l_inf distance
            
        Returns:
            str: Filename of the selected scenario
        """
        candidates = []
        x0, y0 = ignition_point
        
        for (x, y), filenames in self.ignition_map.items():
            if max(abs(x - x0), abs(y - y0)) <= leeway_distance:
                candidates.extend(filenames)
                
        if not candidates:
            raise ValueError(f"No scenario found around {ignition_point} with leeway {leeway_distance}")
        
        return random.choice(candidates)


if __name__ == "__main__":
    import numpy as np
    import os


    test_folder = "MinimalDataset/0002/scenarii"
    os.makedirs(test_folder, exist_ok=True)

    # === Step 2: Initialize the sampler ===
    sampler = ScenarioSampler(test_folder)
    # print(f"Sampler built with {len(sampler.ignition_map)} ignition points.")
    # print(f"Sampled ignition points: {sampler.ignition_map}")
    # === Step 3: Test getting a scenario ===
    try:
        ignition_point = (96, 161)
        leeway = 5
        selected_scenario = sampler.get_scenario_location(ignition_point, leeway)
        print(f"Selected scenario near {ignition_point} (leeway {leeway}): {selected_scenario}")
    except ValueError as e:
        print(f"Error: {e}")