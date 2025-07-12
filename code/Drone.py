# Romain Puech, 2024
# Drone class

import numpy as np
#TODO out of battery not implemented yet
class Drone():
    """
    A class representing a drone in the wildfire detection system.
    
    The drone can move around the grid, detect fires, and recharge at charging stations.
    It maintains its position, battery levels, and state (charging/flying).
    
    Args:
        x (int): Initial x-coordinate
        y (int): Initial y-coordinate
        state (str): Initial state ('charge' or 'fly')
        charging_stations_locations (list): List of (x,y) tuples for charging stations
        N (int): Grid height
        M (int): Grid width
        max_distance_battery (int): Maximum distance the drone can travel before recharging
        max_time_battery (int): Maximum time the drone can fly before recharging
        current_distance_battery (int, optional): Current distance battery level
        current_time_battery (int, optional): Current time battery level
    
    Raises:
        ValueError: If initial position is not at a charging station
    """
    def __init__(self, x, y, state,charging_stations_locations, N, M, max_distance_battery=100, max_time_battery=100, current_distance_battery=None, current_time_battery=None):
        if (x,y) not in charging_stations_locations and [x,y] not in charging_stations_locations:
            raise ValueError("Drone should start on a charging station")
        self.x = x
        self.y = y
        self.N = N
        self.M = M if not(M is None) else N
        self.charging_stations_locations = charging_stations_locations
        self.max_distance_battery = max_distance_battery
        self.max_time_battery = max_time_battery
        self.distance_battery = max_distance_battery if current_distance_battery is None else current_distance_battery
        self.time_battery = max_time_battery if current_time_battery is None else current_time_battery    
        self.state = state
        self.alive = True
        if state == "charge":
            self.distance_battery = self.max_distance_battery
            self.time_battery = self.max_time_battery
    
    def get_position(self):
        """
        Returns the current position of the drone.
        
        Returns:
            tuple: (x,y) coordinates of the drone
        """
        return self.x, self.y
    
    def get_battery(self):
        """
        Returns the current battery levels of the drone.
        
        Returns:
            tuple: (distance_battery, time_battery) levels
        """
        return self.distance_battery, self.time_battery
    
    def get_state(self):
        """
        Returns the current state of the drone.
        
        Returns:
            str: Current state ('charge' or 'fly')
        """
        return self.state

    def is_alive(self):
        """
        Checks if the drone is still operational.
        
        Returns:
            bool: True if drone is operational, False otherwise
        """
        return self.alive
    
    def move(self, dx, dy):
        """
        Moves the drone by the specified delta coordinates.
        
        Args:
            dx (int): Change in x-coordinate
            dy (int): Change in y-coordinate
            
        Returns:
            tuple: (x, y, distance_battery, time_battery, state) after movement
        """
        self.state = "fly"
        self.x += dx
        self.y += dy
        self.x = max(0,min(self.x,self.N-1))
        self.y = max(0,min(self.y,self.M-1))
        self.distance_battery -= (abs(dx) + abs(dy)) # manhathan distance for the moment
        self.time_battery -= 1
        # if not self._check_battery():
        #     print(f"Drone is dead at ({self.x}, {self.y})")
        #     return self.x, self.y, self.distance_battery, self.time_battery, "dead" #TODO FIGURE THIS OUT
        return self.x, self.y, self.distance_battery, self.time_battery, self.state
    
    def fly(self, x,y):
        """
        Makes the drone fly directly to the specified coordinates.
        
        Args:
            x (int): Target x-coordinate
            y (int): Target y-coordinate
            
        Returns:
            tuple: (x, y, distance_battery, time_battery, state) after flying
        """
        self.state = "fly"
        self.x = x
        self.y = y
        self.distance_battery -= (abs(self.x-x) + abs(self.y-y))
        self.time_battery -= 1
        # if not self._check_battery():
        #     print(f"Drone is dead at ({self.x}, {self.y})")
        #     return self.x, self.y, self.distance_battery, self.time_battery, "dead" #TODO FIGURE THIS OUT
        return self.x, self.y, self.distance_battery, self.time_battery, self.state
    
    def recharge(self,x,y):
        """
        Makes the drone recharge at the specified charging station.
        
        Args:
            x (int): Charging station x-coordinate
            y (int): Charging station y-coordinate
            
        Returns:
            tuple: (x, y, distance_battery, time_battery, state) after recharging
        """
        #if (self.x, self.y) in self.charging_stations_locations:#TODO CHECK IF IT IS FROM NEIGHBORING CELL!
        self.x = x
        self.y = y
        self.state = "charge"
        self.distance_battery = self.max_distance_battery
        self.time_battery = self.max_time_battery
        return self.x, self.y, self.distance_battery, self.time_battery, self.state

    
    def route(self, action):
        """
        Executes the specified action for the drone.
        
        Args:
            action (tuple): (action_type, parameters) where:
                - action_type is one of: 'move', 'fly', 'charge'
                - parameters are the coordinates or movement deltas
                
        Returns:
            tuple: (x, y, distance_battery, time_battery, state) after action
            
        Raises:
            ValueError: If action_type is invalid
        """
        if action[0] == 'move':
            return self.move(*action[1])
        elif action[0] == 'fly':
            return self.fly(*action[1])
        elif action[0] == 'charge':
            return self.recharge(*action[1])
        else:
            raise ValueError(f"Invalid action: {action}")

    def _check_battery(self):
        """
        Checks if the drone's battery levels are sufficient.
        
        Returns:
            bool: True if battery levels are sufficient, False otherwise
        """
        #TODO FIGURE THIS OUT
        return True
        if self.time_battery <= 0:
            self.alive = False
            return False
        return True
