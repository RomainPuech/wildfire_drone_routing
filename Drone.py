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

class Drone():
    def __init__(self, x, y, battery=100, routing=default_routing_strategy):
        self.x = x
        self.y = y
        self.battery = battery
        self.routing_strategy = routing
        self.planned_movements = []
        self.fire_alert = False
        self.smoke_alert = False
    
    def get_position(self):
        return self.x, self.y
    
    def move(self, dx, dy):
        self.x += dx
        self.y += dy
        self.battery -= 1

    def route(self, fire_grid, smoke_grid, wind_direction):
        return self.routing_strategy(self, fire_grid, smoke_grid, wind_direction)
