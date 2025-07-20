import random

# ATENÇÃO: Não usei a representação por permutação de string de bits como pede no projeto (10010011)
# Ao invés disso, usei uma representação mais direta como na função gerar_individuo() (71325640)
# Desse jeito, cada posição da lista representa uma coluna e o número em cada coluna seria o número da linha em que a rainha está
# Caso precisemos realmente mudar a representação para uma permutação de string de bits, não terá problema, é MUITO simples
# Bastaria apenas gerar strings de 24 bits, em que cada 3 bits representam um numero de 0 a 7 e fazer a conversão depois
# Ex.: '111001011010101110100000' = '111','001','011','010','101','110','100','000' = [7,1,3,2,5,6,4,0]

def gerar_individuo():
    individuo = list(range(8))  # linhas de 0 a 7
    random.shuffle(individuo)
    return individuo

def gerar_populacao(tamanho=100):
    return [gerar_individuo() for _ in range(tamanho)]

# Essa função apenas calcula o fitness do individuo
# ATENÇÃO: 28 é o fitness perfeito, pois temos 28 pares possíveis entre as 8 rainhas
# Ter fitness = 28 quer dizer que nenhum par de rainhas pode se matar (sem conflitos)
def fitness(individuo):
    conflitos = 0
    for i in range(8):
        for j in range(i + 1, 8):
            if abs(individuo[i] - individuo[j]) == abs(i - j):
                conflitos += 1
    return 28 - conflitos

# Selecionamos 5 individuos aleatoriamente e então, selecionamos 2 desses individuos aleatoriamente e escolhemos o de maior fitness
def selecionar_pai(populacao, fitness_pop, torneio_tamanho=2, candidatos=5):
    # Escolhe 'candidatos' indivíduos aleatórios
    candidatos_indices = random.sample(range(len(populacao)), candidatos)
    
    # Faz torneios de tamanho 2 dentro dos candidatos
    melhor_individuo = None
    melhor_fitness = -float('inf')

    for _ in range(torneio_tamanho):
        ind1, ind2 = random.sample(candidatos_indices, 2)
        
        if fitness_pop[ind1] > fitness_pop[ind2]:
            vencedor = ind1
        else:
            vencedor = ind2
        
        # Atualiza melhor
        if fitness_pop[vencedor] > melhor_fitness:
            melhor_fitness = fitness_pop[vencedor]
            melhor_individuo = vencedor
    
    return populacao[melhor_individuo]

# Essa função apenas serve para evitar escolher o mesmo individuo como o pai1 e pai2 ao mesmo tempo
def selecionar_pais(populacao, fitness_pop):
    pai1 = selecionar_pai(populacao, fitness_pop)
    while True:
        pai2 = selecionar_pai(populacao, fitness_pop)
        if pai2 != pai1:
            break
    return pai1, pai2

# Função de crossing-over com probabilidade de permutação de 90%
def cut_and_crossfill(pai1, pai2, probabilidade=0.9):
    # Decide se aplica o crossover
    if random.random() > probabilidade:
        return pai1[:], pai2[:]  # Cópias exatas dos pais

    n = len(pai1)
    corte = random.randint(1, n - 1)

    # Filho 1: prefixo do pai1 + resto do pai2 sem repetição
    filho1 = pai1[:corte]
    for gene in pai2:
        if gene not in filho1:
            filho1.append(gene)

    # Filho 2: prefixo do pai2 + resto do pai1 sem repetição
    filho2 = pai2[:corte]
    for gene in pai1:
        if gene not in filho2:
            filho2.append(gene)

    return filho1, filho2

# Função de mutação com probabilidade de 40%
def mutar(individuo, probabilidade=0.4):
    if random.random() < probabilidade:
        i, j = random.sample(range(len(individuo)), 2)
        individuo[i], individuo[j] = individuo[j], individuo[i]
    return individuo

def substituir_piores(populacao, fitness_pop, filhos, fitness_filhos):
    # Índices dos dois piores indivíduos
    piores_indices = sorted(range(len(fitness_pop)), key=lambda i: fitness_pop[i])[:2]

    for filho, fit in zip(filhos, fitness_filhos):
        # Pega o pior atual
        pior_idx = piores_indices.pop(0)
        if fit > fitness_pop[pior_idx]:
            populacao[pior_idx] = filho
            fitness_pop[pior_idx] = fit
        # Senão: o filho é descartado (não substitui ninguém)

    return populacao, fitness_pop

# Essa função roda no máximo 10.000 vezes, podendo encerrar antes se encontrar a solução
def solve():
    populacao = gerar_populacao()
    fitness_pop = []
    
    for individuo in populacao:
        fitness_pop.append(fitness(individuo))
        
    tentativa = 0
    
    while tentativa < 10000 and max(fitness_pop) < 28:
        pai1, pai2 = selecionar_pais(populacao, fitness_pop)
        filho1, filho2 = cut_and_crossfill(pai1, pai2)
        
        filho1 = mutar(filho1)
        filho2 = mutar(filho2)
        
        filhos = [filho1, filho2]
        fitness_filhos = [fitness(filho1), fitness(filho2)]
        
        populacao, fitness_pop = substituir_piores(populacao, fitness_pop, filhos, fitness_filhos)
        
        tentativa += 1
        
    melhor_idx = fitness_pop.index(max(fitness_pop))
    return populacao[melhor_idx], tentativa

resolucao, tentativa = solve()
print(f"Resolução: {resolucao} | Tentativa: {tentativa}")
