import random
from typing import List
import statistics

class Gene:
    def __init__(self, decimal_number:int = None, bit1:int = None, bit2:int = None, bit3:int = None):
        if decimal_number is not None:
            self.bits = self.set_number10(decimal_number)
        else:
            self.bits = [bit1, bit2, bit3]
            for i in range(len(self.bits)):
                if self.bits[i] is None:
                    self.bits[i] = random.randint(0,1)

    def __str__(self):
        return f'{self.bits}'
    
    def __repr__(self):
        return f'{self.bits}'
    
    def __eq__(self, other):
        if not isinstance(other, Gene):
            return NotImplemented
        return self.bits==other.bits

    def set_number10(self, num:int):
        binary_number = []
        for i in range(2, -1, -1):
            binary_number.append(num//(2**i))
            num = num%(2**i)
        return binary_number

    def get_number10(self):
        sum = 0
        for i in range(len(self.bits)):
            sum += self.bits[i] * (2**(len(self.bits)-1-i))
        return sum

class Individual:
    def __init__(self, num_genes:int, vector_genes:List[Gene] = None):
        self.num_genes = num_genes
        if vector_genes is None:
            individual_genes = list(range(num_genes))
            random.shuffle(individual_genes)
            self.genes = [Gene(num) for num in individual_genes]
        else:
            self.genes = vector_genes 
            
        self.fitness = self.calculate_fitness()
    
    def __str__(self):
        return f'{[gene.get_number10() for gene in self.genes]}'
    
    def __repr__(self):
        return self.__str__()#f'{[gene.get_number10() for gene in self.genes]}'

    def print_genes_bits(self):
        print(self.genes)
    
    # Essa função apenas calcula o fitness do individuo
    # ATENÇÃO: 28 é o fitness perfeito, pois temos 28 pares possíveis entre as 8 rainhas
    # Ter fitness = 28 quer dizer que nenhum par de rainhas pode se matar (sem conflitos)
    def calculate_fitness(self):
        conflitos = 0
        for i in range(8):
            for j in range(i + 1, 8):
                if abs(self.genes[i].get_number10() - self.genes[j].get_number10()) == abs(i - j):
                    conflitos += 1
        return 28 - conflitos
    
    # Função de mutação com probabilidade de 40%
    def mutar(self, probabilidade:float=0.4):
        if random.random() < probabilidade:
            i, j = random.sample(range(len(self.genes)), 2)
            self.genes[i], self.genes[j] = self.genes[j], self.genes[i]
        self.fitness = self.calculate_fitness()
        
    # Função de crossing-over com probabilidade de permutação de 90%
    def cut_and_crossfill(self, parceiro:"Individual", probabilidade=0.9):
        # Decide se aplica o crossover
        if random.random() > probabilidade:
            return self, parceiro  # Cópias exatas dos pais

        n = self.num_genes
        corte = random.randint(1, n - 1)

        # Filho 1: prefixo do pai1 + resto do pai2 sem repetição
        vector_filho1 = self.genes[:corte]
        for gene in parceiro.genes:
            if gene not in vector_filho1:
                vector_filho1.append(gene)

        # Filho 2: prefixo do pai2 + resto do pai1 sem repetição
        vector_filho2 = parceiro.genes[:corte]
        for gene in self.genes:
            if gene not in vector_filho2:
                vector_filho2.append(gene)

        filho1 = Individual(self.num_genes, vector_filho1)
        filho2 = Individual(parceiro.num_genes, vector_filho2)
        return filho1, filho2

        
class Populacao:
    def __init__(self, tamanho:int, num_genes:int = 8):
        self.tamanho = tamanho
        self.individuos = [Individual(num_genes) for _ in range(tamanho)]
        self.all_mean = []
        self.all_stdev = []
        
    def selecionar_pais(self, candidatos:int=5):
        # Escolhe 'candidatos' indivíduos aleatórios
        candidatos_indices = random.sample(range(self.tamanho), candidatos)
        candidatos_sort = sorted(candidatos_indices, key = lambda i:self.individuos[i].fitness)
        pais = candidatos_sort[candidatos-2:]
        return self.individuos[pais[0]], self.individuos[pais[1]]
    
    def substituir_piores(self, filhos:list):
        # Índices dos dois piores indivíduos
        piores_indices = sorted(range(len(self.individuos)), key=lambda i: self.individuos[i].fitness)[:2]

        for filho in filhos:
            # Pega o pior atual
            pior_idx = piores_indices.pop(0)
            if filho.fitness > self.individuos[pior_idx].fitness:
                self.individuos[pior_idx] = filho
            # Senão: o filho é descartado (não substitui ninguém)
            
    def melhor_individuo(self):
        return max(self.individuos, key=lambda x: x.fitness)
    
    def fitness_media(self):
        return sum([individuo.fitness for individuo in self.individuos])/len(self.individuos)
    
    def fitness_stdev(self):
        return statistics.stdev([individuo.fitness for individuo in self.individuos])
    
    # Essa função roda no máximo 10.000 vezes, podendo encerrar antes se encontrar a solução
    def solve(self, max_tentativa:int = 10000, candidatos_pai:int = 5, prob_permutacao:float = 0.9, prob_mutacao:float = 0.4):
        tentativa = 0
        while tentativa < max_tentativa and self.melhor_individuo().fitness < 28:
            pai1, pai2 = self.selecionar_pais(candidatos_pai)
            filho1, filho2 = pai1.cut_and_crossfill(pai2, prob_permutacao)
            
            filho1.mutar(prob_mutacao)
            filho2.mutar(prob_mutacao)
            
            filhos = [filho1, filho2]
            
            self.substituir_piores(filhos)
            
            self.all_mean.append(self.fitness_media())
            self.all_stdev.append(self.fitness_stdev())
            tentativa += 1
            
        return self.melhor_individuo(), tentativa