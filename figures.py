import numpy as np


class Figure:
    def is_valid(self, dices:np.array):
        return NotImplementedError

    def compute_value(self, dices:np.array):
        return NotImplementedError

    def get_possible_values(self , n_dice:int, n_face:int):
        return NotImplementedError
    
    def get_max_values(self, n_dice:int, n_face:int):
        return NotImplementedError
    
class Chance(Figure):
    def is_valid(self, dices:np.array):
        return True

    def compute_value(self, dices:np.array):
        return (dices * (np.arange(dices.shape[0]) + 1)).sum()
    
    def get_possible_values(self, n_dice:int, n_face:int):
        return [i for i in range(n_dice, (n_dice * (n_face + 1)))]
    
    def get_max_values(self, n_dice:int, n_face:int):
        return n_dice * (n_face + 1)
    

class Number(Figure):
    def __init__(self, number:int):
        self.number = number

    def is_valid(self, dices:np.array):
        return dices[self.number] > 0

    def compute_value(self, dices:np.array):
        return (int(self.number + 1) * dices[self.number])

    def get_possible_values(self, n_dice:int, n_face:int):
        return [int((self.number + 1) * i) for i in range(n_dice+1)]
        
    def get_max_values(self, n_dice:int, n_face:int):
        return int(n_dice * (self.number + 1))
        
class Suite(Figure):
    def __init__(self, start:int, end:int, value:int=30):
        self.start = start
        self.end = end
        self.value = value
    
    def is_valid(self, dices):
        return np.min(dices[self.start:self.end]) > 0
    
    def compute_value(self, dices):
        if not self.is_valid(dices):
            return 0
        return self.value
    def get_possible_values(self, n_dice:int, n_face:int):
        return [0, self.value]
    
    def get_max_values(self, n_dice:int, n_face:int):
        return self.value
    
class Brelan(Figure):
    def is_valid(self, dices:np.array):
        return bool(np.max(dices) >= 3)

    def compute_value(self, dices:np.array):
        if not self.is_valid(dices):
            return 0
        return int((np.argmax(dices) + 1) * 3)
    
    def get_possible_values(self, n_dice:int, n_face:int):
        return [i*3 for i in range(n_face + 1)]
    
    def get_max_values(self, n_dice:int, n_face:int):
        return 3 * n_face
    
class Multiple(Figure):
    def __init__(self, nb:int, value:int):
        self.nb = nb
        self.value = value

    def is_valid(self, dices:np.array):
        return np.max(dices) >= self.nb
    
    def compute_value(self, dices:np.array):
        if not self.is_valid(dices):
            return 0
        return self.value
    def get_possible_values(self, n_dice:int, n_face:int):
        return [0, self.value]
    
    def get_max_values(self, n_dice:int, n_face:int):   
        return self.value
    
class Full(Figure):
    def is_valid(self, dices:np.array):
        result = np.sort(dices)
        return result[-1] == 3 and result[-2] == 2

    def compute_value(self, dices:np.array):
        if not self.is_valid(dices):
            return 0
        return 25
    def get_possible_values(self, n_dice:int, n_face:int):
        return [0, 25]
    
    def get_max_values(self, n_dice:int, n_face:int):
        return 25

if __name__ == "__main__":
    dices = np.bincount(np.random.randint(0, 6, (5)), minlength=6)
    print(dices)
    print(Brelan().is_valid(dices))
    print(Brelan().compute_value(dices))
    print(Full().is_valid(dices))
    print(Full().compute_value(dices))
    print(Number(2).is_valid(dices))
    print(Number(2).compute_value(dices))
    print(Number(5).is_valid(dices))
    print(Number(5).compute_value(dices))