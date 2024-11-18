class Box:
    def __init__(self, coords, time):
        self.coords = coords  
        self.time = time      

class MovingObject:
    def __init__(self, starting_box):
        self.boxes = [starting_box]

    def add_box(self, box):
        self.boxes.append(box)

    def last_coords(self):
        return self.boxes[-1].coords

    def age(self, curr_time):
        last_time = self.boxes[-1].time
        return curr_time - last_time
