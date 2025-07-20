import random
from typing import List
import statistics
import numpy as np
import copy

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
            self.genes = copy.deepcopy(vector_genes)
            
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
    def calculate_fitness_old(self):
        conflitos = 0
        for i in range(8):
            for j in range(i + 1, 8):
                if abs(self.genes[i].get_number10() - self.genes[j].get_number10()) == abs(i - j):
                    conflitos += 1
        return 28 - conflitos
    
    # Função nova de cálculo de fitness
    # Calcula o número de pares sem conflitos - mais pares -> tabuleiro mais ideal
    def calculate_fitness(self):
        pares_sem_conflito = 0
        for i in range(len(self.genes)):
            for j in range(i+1,len(self.genes)):
                diagonal = abs(self.genes[i].get_number10() - self.genes[j].get_number10())
                if diagonal != abs(i - j):
                    pares_sem_conflito += 1
        return pares_sem_conflito
    
    # Generalizada a lógica de verificação de solução
    def valid_solution(self):
        for i in range(8):
            for j in range(i + 1, 8):
                if abs(self.genes[i].get_number10() - self.genes[j].get_number10()) == abs(i - j):
                    return False
        return True
    
    # Função de mutação com probabilidade de 40%
    def mutar_old(self, probabilidade:float=0.4):
        if random.random() < probabilidade:
            i, j = random.sample(range(len(self.genes)), 2)
            self.genes[i], self.genes[j] = self.genes[j], self.genes[i]
        self.fitness = self.calculate_fitness()
    
    #Mutação por inserção
    def mutar(self, probabilidade:float=0.4):
        if random.random() < probabilidade:
            i, j = random.sample(range(len(self.genes)), 2)
            if i > j:
                i, j = j, i
            self.genes = self.genes[:i] + [self.genes[i]] + [self.genes[j]] + self.genes [i+1:j] + self.genes[j+1:]
        self.fitness = self.calculate_fitness()
        
    # Função de crossing-over com probabilidade de permutação de 90%
    def cut_and_crossfill(self, parceiro:"Individual", probabilidade=0.9):
        # Decide se aplica o crossover
        if random.random() > probabilidade:
            filho1 = Individual(self.num_genes, self.genes[:])
            filho2 = Individual(parceiro.num_genes, parceiro.genes[:])
            return filho1, filho2  # Cópias exatas dos pais

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
    
    def border_recombination(self, parceiro:"Individual", probabilidade=0.9, quantidade_filhos=2):
        # Decide se aplica o crossover
        if random.random() > probabilidade:
            filho1 = Individual(self.num_genes, self.genes[:])
            filho2 = Individual(parceiro.num_genes, parceiro.genes[:])
            return filho1, filho2  # Cópias exatas dos pais
        
        dict_border = dict()
        for i in range(self.num_genes):
            dict_border[i] = []
            
        for gene_idx in range(len(self.genes)):
            dict_border[self.genes[i].get_number10()].append(self.genes[(gene_idx-1)%len(self.genes)])
            dict_border[self.genes[i].get_number10()].append(self.genes[(gene_idx+1)%len(self.genes)])
            dict_border[self.genes[i].get_number10()].append(parceiro.genes[(gene_idx-1)%len(self.genes)])
            dict_border[self.genes[i].get_number10()].append(parceiro.genes[(gene_idx+1)%len(self.genes)])
        
        vetor_filhos = []
        for _ in range(quantidade_filhos):
            filho_atual = []
            gene_atual = random.choice(self.genes)
            filho_atual.append(gene_atual)
            while len(filho_atual) < len(self.genes):
                vizinhos_validos = [x for x in dict_border[gene_atual.get_number10()] if x not in filho_atual]
                seen = set()
                dupes = {x.get_number10() for x in vizinhos_validos if x.get_number10() in seen or seen.add(x.get_number10())}
                if(len(dupes) > 0):
                    num_atual = random.choice(list(dupes))
                    gene_atual = Gene(num_atual)
                    filho_atual.append(gene_atual)
                elif len(vizinhos_validos)>1:
                    menor_tamanho = min(map(lambda i: len(dict_border[i.get_number10()]), vizinhos_validos))
                    menores = [vizinho for vizinho in vizinhos_validos if len(dict_border[vizinho.get_number_10()]) == menor_tamanho]
                    gene_atual = random.choice(menores)
                    filho_atual.append(gene_atual)
                else:
                    restantes = [g for g in self.genes if g not in filho_atual]
                    gene_atual = random.choice(restantes)
                    filho_atual.append(gene_atual)
            vetor_filhos.append(Individual(self.num_genes, filho_atual))
        return vetor_filhos
        

        
class Populacao:
    def __init__(self, tamanho:int, num_genes:int = 8):
        self.tamanho = tamanho
        self.individuos = [Individual(num_genes) for _ in range(tamanho)]
        self.all_mean = []
        self.all_stdev = []
        self.all_best_ind = []
        
    def selecionar_pais(self, candidatos:int=5):
        # Escolhe 'candidatos' indivíduos aleatórios
        candidatos_indices = random.sample(range(self.tamanho), candidatos)
        candidatos_sort = sorted(candidatos_indices, key = lambda i:self.individuos[i].fitness)
        pais = candidatos_sort[candidatos-2:]
        return self.individuos[pais[0]], self.individuos[pais[1]]
    
    def selecionar_pais_roulette(self, candidatos:int=2):
        # Seleção por roleta
        total_fitness = sum(ind.fitness for ind in self.individuos)
        if total_fitness == 0:
            return random.choice(self.individuos), random.choice(self.individuos)

        roleta = [ind.fitness / total_fitness for ind in self.individuos]
        roleta_acumulada = [sum(roleta[:i+1]) for i in range(len(roleta))]

        pais_indices = []
        for _ in range(candidatos):
            r = random.random()
            for i, valor in enumerate(roleta_acumulada):
                if r <= valor:
                    pais_indices.append(i)
                    break

        pais_indices = sorted(pais_indices, key=lambda i: self.individuos[i].fitness, reverse=True)
        return self.individuos[pais_indices[0]], self.individuos[pais_indices[1]]

    
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
    
    def pior_individuo(self):
        return min(self.individuos, key=lambda x: x.fitness)
    
    def fitness_media(self):
        return sum([individuo.fitness for individuo in self.individuos])/len(self.individuos)
    
    def fitness_stdev(self):
        return statistics.stdev([individuo.fitness for individuo in self.individuos])
    
    def max_fit_ind(self):
        return [ind for ind in self.individuos if ind.valid_solution()]
    
    # Essa função roda no máximo 10.000 vezes, podendo encerrar antes se encontrar a solução
    def solve(self, max_tentativa:int = 10000, candidatos_pai:int = 5, prob_permutacao:float = 0.9, prob_mutacao:float = 0.4):
        tentativa = 0
        while tentativa < max_tentativa and not self.melhor_individuo().valid_solution():
            pai1, pai2 = self.selecionar_pais_roulette(candidatos_pai)
            
            filho1, filho2 = pai1.cut_and_crossfill(pai2, prob_permutacao)
            
            filho1.mutar(prob_mutacao)
            filho2.mutar(prob_mutacao)
            
            filhos = [filho1, filho2]
            
            self.substituir_piores(filhos)
            
            self.all_mean.append(self.fitness_media())
            self.all_stdev.append(self.fitness_stdev())
            self.all_best_ind.append(self.melhor_individuo().fitness)
            tentativa += 1
            
        return self.melhor_individuo(), tentativa
    
    def solve_all(self, candidatos_pai:int = 5, prob_permutacao:float = 0.9, prob_mutacao:float = 0.4):
        tentativa = 0
        while not self.pior_individuo().valid_solution():
            pai1, pai2 = self.selecionar_pais_roulette(candidatos_pai)
            
            filho1, filho2 = pai1.cut_and_crossfill(pai2, prob_permutacao)
            
            filho1.mutar(prob_mutacao)
            filho2.mutar(prob_mutacao)
            
            filhos = [filho1, filho2]
            
            self.substituir_piores(filhos)
            
            self.all_mean.append(self.fitness_media())
            self.all_stdev.append(self.fitness_stdev())
            self.all_best_ind.append(self.melhor_individuo().fitness)
            tentativa += 1

            
        return self.melhor_individuo(), tentativa
    
    
if __name__ == "__main__":
    pai1v = [Gene(1), Gene(2), Gene(3), Gene(4), Gene(5), Gene(6), Gene(7), Gene(0)]
    pai2v = [Gene(3), Gene(7), Gene(0), Gene(2), Gene(6), Gene(5), Gene(1), Gene(4)]
    pai1 = Individual(8,pai1v)
    pai2 = Individual(8,pai2v)
    filho1, filho2 = pai1.border_recombination(pai2)
    print(filho1, filho2)