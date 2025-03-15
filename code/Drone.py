# Romain Puech, 2024
# Drone class

import numpy as np
#TODO out of battery not implemented yet
class Drone():
    def __init__(self, x, y, charging_stations_locations, N, M, distance_battery=100, time_battery=100):
        if (x,y) not in charging_stations_locations and [x,y] not in charging_stations_locations:
            raise ValueError("Drone should start on a charging station")
        self.x = x
        self.y = y
        self.N = N
        self.M = M
        self.charging_stations_locations = charging_stations_locations
        self.distance_battery = distance_battery
        self.max_distance_battery = distance_battery
        self.time_battery = time_battery
        self.max_time_battery = time_battery
        self.state = "fly" # maybe should start with "charge"?
    
    def get_position(self):
        return self.x, self.y
    
    def get_battery(self):
        return self.distance_battery, self.time_battery
    
    def get_state(self):
        return self.state
    
    def move(self, dx, dy):
        self.state = "fly"
        self.x += dx
        self.y += dy
        self.x = max(0,min(self.x,self.N-1))
        self.y = max(0,min(self.y,self.M-1))
        self.distance_battery -= (abs(dx) + abs(dy)) # manhathan distance for the moment
        self.time_battery -= 1
        return self.x, self.y, self.distance_battery, self.time_battery, self.state
    
    def fly(self, x,y):
        self.state = "fly"
        self.x = x
        self.y = y
        self.distance_battery -= (abs(self.x-x) + abs(self.y-y))
        self.time_battery -= 1
        return self.x, self.y, self.distance_battery, self.time_battery, self.state
    
    def recharge(self):
        if (self.x, self.y) in self.charging_stations_locations:
            self.state = "charge"
            self.distance_battery = self.max_distance_battery
            self.time_battery = self.max_time_battery
        return self.x, self.y, self.distance_battery, self.time_battery, self.state

    
    def route(self, action):
        if action[0] == 'move':
            return self.move(*action[1])
        elif action[0] == 'fly':
            return self.fly(*action[1])
        elif action[0] == 'charge':
            return self.recharge()

