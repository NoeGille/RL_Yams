import numpy as np

class Dice:
    '''Class to represent a dice'''
    def __init__(self, nb_faces:int):
        self.nb_faces = nb_faces
        self.value = None
        self.roll()
    
    def roll(self):
        self.value = np.random.randint(self.nb_faces) + 1
    
    def __repr__(self):
        return str(f'Dice{self.value}')

class Figure:
    def is_valid(self, dices:list[Dice]):
        return NotImplementedError

    def compute_value(self, dices:list[Dice]):
        return NotImplementedError

class Chance(Figure):
    def is_valid(self, dices:list[Dice]):
        return True

    def compute_value(self, dices:list[Dice]):
        return sum([d.value for d in dices])

class Number(Figure):
    def __init__(self, number:int):
        self.number = number

    def is_valid(self, dices:list[Dice]):
        return self.number in map(lambda x: x.value, dices)

    def compute_value(self, dices:list[Dice]):
        return sum([d.value for d in dices if d.value == self.number])

class Suite(Figure):
    def __init__(self, start:int, end:int, value:int=30):
        self.start = start
        self.end = end
        self.value = value
    
    def is_valid(self, dices):
        values = [d.value for d in dices]
        for i in range(self.start, self.end):
            if i not in values:
                return False
        return True
    
    def compute_value(self, dices):
        if not self.is_valid(dices):
            return 0
        return self.value
    
class Brelan(Figure):
    def is_valid(self, dices:list[Dice]):
        result = np.bincount([d.value for d in dices])
        return result.max() >= 3

    def compute_value(self, dices:list[Dice]):
        if not self.is_valid(dices):
            return 0
        return sum([d.value for d in dices])

class Multiple(Figure):
    def __init__(self, nb:int, value:int):
        self.nb = nb
        self.value = value

    def is_valid(self, dices:list[Dice]):
        result = np.bincount([d.value for d in dices])
        return result.max() >= self.nb
    
    def compute_value(self, dices:list[Dice]):
        if not self.is_valid(dices):
            return 0
        return self.value

class Full(Figure):
    def is_valid(self, dices:list[Dice]):
        result = np.bincount([d.value for d in dices])
        result = np.sort(result)
        return result[-1] == 3 and result[-2] == 2

    def compute_value(self, dices:list[Dice]):
        if not self.is_valid(dices):
            return 0
        return 25

if __name__ == "__main__":
    dices = [Dice(6) for _ in range(5)]
    print(dices)
    print(Brelan().is_valid(dices))
    print(Brelan().compute_value(dices))
    print(Full().is_valid(dices))
    print(Full().compute_value(dices))
    print(Number(3).is_valid(dices))
    print(Number(3).compute_value(dices))
    print(Number(6).is_valid(dices))
    print(Number(6).compute_value(dices))