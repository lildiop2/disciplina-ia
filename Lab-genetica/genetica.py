import numpy as np
import math
from typing import Callable, Tuple
import matplotlib.pyplot as plt

FEATURE_BIT_SIZE: int = 18
FEATURE_RANGE: float = 20.

class GeneticAlgorithm:
    # class Individual: describes the problem's solutions
    class Individual:
        def __init__(self, x: str = '', y: str = ''):
            self.x = x
            self.y = y
            self.eval = 0
            self.fitness = 0
        
        def array(self) -> []:
            return [self.x, self.y, self.eval, self.fitness, self]

    def __init__(self, popsize: int = 10, n_elite: int = 10, max_it: float = 100, crossover_threshold: float = 0.7,
                    mutation_threshold: float = 0.005, min_diff: float = .01, num_res_repetition: int = 3, seed: int = 1):
        np.random.seed(seed)

        self.popsize = popsize
        self.max_it = max_it
        self.crossover_threshold = crossover_threshold
        self.mutation_threshold = mutation_threshold
        self.min_diff = min_diff
        self.n_elite = round(n_elite/2)*2 # make sure it's even
        self.num_res_repetition = num_res_repetition

        self.individuals = self.generate_individuals()
        self.best = None
        self.past_best = None
    
    # Generates random binary number (as a string)
    @classmethod
    def gen_bin(self) -> str:
        bin_nmb = ''
        for i in range(FEATURE_BIT_SIZE):
            bin_nmb += str(np.random.randint(2))
        
        return bin_nmb #if self.bin2float(bin_nmb) > 0 else '000000000000000000'
    
    # Generate random individuals
    def generate_individuals(self) -> np.array:
        arr = []
        for i in range(self.popsize):
            arr.append(self.Individual(self.gen_bin(), self.gen_bin()))
        return arr

    # Convert binary number to a float/real range
    @classmethod
    def bin2float(self, x: str) -> float: # using 4 decimal places with range 20 -> needs 18 bits
        i_x = int(x, 2)
        return i_x *  (FEATURE_RANGE) / (math.pow(2, FEATURE_BIT_SIZE)-1) + (-FEATURE_RANGE/2)

    # Objective method: Alpine function
    def obj_function(self, ind: Individual) -> float:
        x, y = self.bin2float(ind.x), self.bin2float(ind.y)
        #print(f"ind: {x}, {y}")
        return - (math.sqrt(abs(x))*math.sin(abs(x)) ) * (math.sqrt(abs(y))*math.sin(abs(y)))
    
    # Fitness method: Linear ranking
    def fit(self, r_max: float, r_min: float, index: int) -> float:
        return r_min + (r_max - r_min) * (self.popsize - index) / (self.popsize - 1)

    # Selection method: 
    def select(self, list_individuals: list) -> Tuple[Individual, list]:
        # Sum vector
        arr_sum = []
        _sum = 0

        # offset to alway return something in wheel
        offset_if_negative_minimum = 2 * abs(list_individuals[0].fitness) if list_individuals[0].fitness < 0 else 0
        for ind in list_individuals:
            _sum += ind.fitness + offset_if_negative_minimum
            arr_sum.append(_sum)
        
        # Select from wheel
        rand_value = np.random.rand() * arr_sum[-1]
        for i, value in enumerate(arr_sum):
            if rand_value <= value:
                e = list_individuals.pop(i)
                return e, list_individuals
        raise Exception("select outside bounds")

    # Crossover method: uniform
    def crossover(self, a: Individual, b: Individual) -> Tuple[Individual, Individual]: # needs CHANCE of generating children
        if np.random.rand() >= self.crossover_threshold:
            return a, b # 30% of not generating any children

        child_a = self.Individual()
        child_b = self.Individual()
        
        uniform_mask = self.gen_bin()
        for i, mask in enumerate(uniform_mask):
            child_a.x += a.x[i] if mask == '0' else b.x[i]
            child_b.x += a.x[i] if mask == '1' else b.x[i]
        
        uniform_mask = self.gen_bin()
        for i, mask in enumerate(uniform_mask):
            child_a.y += a.y[i] if mask == '0' else b.y[i]
            child_b.y += a.y[i] if mask == '1' else b.y[i]
        
        return child_a, child_b
    
    # Mutation method: per bit probability mutation
    def mutate(self, a: Individual) -> Individual:
        mutated_x = ''
        mutated_y = ''
        for bit in a.x:
            mutated_x += bit if np.random.rand() < 1 - self.mutation_threshold else str(int( not bool(int(bit)) )) # 0.5% chance of mutating
        for bit in a.y:
            mutated_y += bit if np.random.rand() < 1 - self.mutation_threshold else str(int( not bool(int(bit)) )) # 0.5% chance of mutating
        a.x = mutated_x
        a.y = mutated_y
        return a
    
#grafivo
    def graph(self):
        X=[]
        Y=[]
        F=[]
        for i in self.individuals:
            X.append(self.bin2float(i.x))
            Y.append(self.bin2float(i.y))
            F.append(self.obj_function(self.Individual(i.x, i.y)))

    # Criando a figura e projeção em 3D
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        ax.plot_trisurf(X, Y, F, linewidth=0, antialiased=True, cmap='jet')
        plt.show()


    # Run method: execute genetic algorithm
    def run(self) -> None:
        it = 0
        same_count = 0
        while not self.best or not self.past_best or (same_count <= self.num_res_repetition and it < self.max_it):
            # CALCULATE RESULTS
            r_max = float('-inf')
            r_min = float('inf')
            ordered = [] # array to sort results: [x, y, eval, fitness, individual]
            for ind in self.individuals:
                x, y = self.bin2float(ind.x), self.bin2float(ind.y)
                # print(f"ind: {x}, {y}")
                ind.eval = self.obj_function(ind)
                if r_max < ind.eval:
                    r_max = ind.eval
                if r_min > ind.eval:
                    r_min = ind.eval
                ordered.append(ind.array())
            # ascending order by result
            ordered = np.array(ordered)
            ordered = ordered[np.argsort(ordered[:, 2])] # min value receives priority on linear ranking
            
            # ASSESS FITNESS
            for i, row in enumerate(ordered):
                ind = row[4]
                ind.fitness = self.fit(r_max, r_min, i)
                ordered[i][3] = ind.fitness 

            ordered = ordered[np.argsort(ordered[:, 3])] # ascending order by fitness (3)
            next_gen = ordered[-self.n_elite:][:,-1].tolist() # extract n_elite(s) then extract only the individuals reference
            self.past_best = self.best
            self.best = ordered[-1:][:,-1][0] # extract best result thus far
            ordered = ordered[:-2] # remove selected elite(s)
            
            ordered = ordered[:,-1].tolist() # transform into list of individuals
            for i in range(round((self.popsize-self.n_elite)/2)):
                parent_a, ordered = self.select(ordered)
                parent_b, ordered = self.select(ordered)
                child_a, child_b = self.crossover(parent_a, parent_b)
                child_a, child_b = self.mutate(child_a), self.mutate(child_b)
                next_gen.append(child_a)
                next_gen.append(child_b)
            
            self.individuals = next_gen
            it += 1
            if self.best and self.past_best and abs(self.past_best.fitness - self.best.fitness) <= self.min_diff:
                same_count += 1
            else:
                same_count = 0
        
        print(f"iteracoes: {it}")
        return self.best

if __name__ == '__main__':
    g = GeneticAlgorithm(n_elite=10, popsize=500, max_it=2000)
    best = g.run()
    print(f"x = {GeneticAlgorithm.bin2float(best.x)}, y = {GeneticAlgorithm.bin2float(best.y)}, f(x,y) = {best.eval}")
    g.graph()

    
    # Num of elites: 5
    # Population size: 500
    # Selection type: roleta
    # Crossover type: uniform
    # Objective function: alpine function
    # Fitness function: linear ranking
    # Max number of generations: 2000
    # Crossover threshold: 70% to generate children
    # Mutation threshold: 0.5% of mutation occurrence per bit