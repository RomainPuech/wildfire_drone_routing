# Romain Puech, 2024
# Drone class and base routing function

import numpy as np

def default_routing_strategy(drone, fire_grid, smoke_grid, wind_direction):
    # define here your routing algorithm
    # if no smoke is detected in neighboring cells, move randomly.
    # if smoke is detected, move in the direction opposite to the wind
    N = fire_grid.shape[0]
    x,y = drone.get_position()
    is_smoke = False
    is_fire = False
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < N: # always use small threshold instead of 0
                if fire_grid[nx, ny] == 1:
                    is_fire = True
                if smoke_grid[nx, ny] > 1e-3:
                    is_smoke = True
                break
    if is_fire:
        drone.fire_alert = True
        drone.planned_movements = []
        return
    drone.fire_alert = False

    if is_smoke:
        drone.smoke_alert = True
        if "up" in wind_direction:
            drone.move(2,0)
        elif "down" in wind_direction:
            drone.move(-2,0)
        if "left" in wind_direction:
            drone.move(0,2)
        elif "right" in wind_direction:
            drone.move(0,-2)
    else:
        drone.smoke_alert = False
        if len(drone.planned_movements) == 0:
            dx, dy = np.random.randint(-1, 2, 2)*2
            drone.planned_movements = [(dx, dy)] * 5
        drone.move(*drone.planned_movements.pop(0))


#TODO out of battery not implemented yet
class Drone():
    def __init__(self, x, y, charging_stations_locations, N, time_battery=100, distance_battery=100):
        if (x,y) not in charging_stations_locations and [x,y] not in charging_stations_locations:
            raise ValueError("Drone should start on a charging station")
        self.x = x
        self.y = y
        self.N = N
        self.charging_stations_locations = charging_stations_locations
        self.time_battery = time_battery
        self.distance_battery = distance_battery
        self.max_time_battery = self.time_battery
        self.max_distance_battery = self.distance_battery
        self.fire_alert = False
        self.smoke_alert = False

    
    def get_position(self):
        return self.x, self.y
    
    def get_battery(self):
        return self.distance_battery, self.time_battery
    
    def move(self, dx, dy):
        self.x += dx
        self.y += dy
        self.x = max(0,min(self.x,self.N-1))
        self.y = max(0,min(self.y,self.N-1))
        self.distance_battery -= (dx+dy) # manhathan distance for the moment
        self.time_battery-=1
        return self.x, self.y, self.distance_battery, self.time_battery
    
    def recharge(self):
        if (self.x, self.y) in self.charging_stations_locations:
            self.time_battery = self.max_time_battery
            self.distance_battery = self.max_distance_battery

        return self.x, self.y, self.distance_battery, self.time_battery

    
    def route(self, action):
        if action[0] == 'move':
            return self.move(*action[1])
        elif action[0] == 'recharge':
            return self.recharge()

