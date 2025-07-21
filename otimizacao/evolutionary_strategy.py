import numpy as np
from benchmark_functions import *
import matplotlib.pyplot as plt

def inicializar_populacao(funcao, pop_size, d):
    fn_name = funcao.__name__
    min_limit = globals()[f"{fn_name.upper()}_MIN_LIMIT"]
    max_limit = globals()[f"{fn_name.upper()}_MAX_LIMIT"]
    return np.random.uniform(min_limit, max_limit, (pop_size, d))

def calcular_sigma_inicial(funcao, sigma_inicial=None):
    if sigma_inicial is not None:
        return sigma_inicial
    fn_name = funcao.__name__
    min_limit = globals()[f"{fn_name.upper()}_MIN_LIMIT"]
    max_limit = globals()[f"{fn_name.upper()}_MAX_LIMIT"]
    return (max_limit - min_limit)/10

def selecionar_elite(pop, fitness, elite_size):
    elite_indices = np.argpartition(fitness, elite_size)[:elite_size]
    return pop[elite_indices]

def recombinar(parents, weights=[0.7, 0.3]):
    return weights[0]*parents[0] + weights[1]*parents[1]

def mutar(individuo, sigma, min_limit, max_limit):
    mutado = individuo + sigma * np.random.randn(len(individuo))
    return np.clip(mutado, min_limit, max_limit)

def atualizar_melhor_solucao(pop, fitness, best_X, best_fitness):
    current_best_idx = np.argmin(fitness)
    if fitness[current_best_idx] < best_fitness:
        return pop[current_best_idx].copy(), fitness[current_best_idx]
    return best_X, best_fitness

def ajustar_sigma(sigma, melhorou):
    if np.random.rand() < 0.2:  
        return sigma * 1.2 if melhorou else sigma * 0.9
    return sigma

def estrategia_evolutiva_modular(
    funcao,
    d: int = 30,
    max_iter: int = 1000,
    pop_size: int = 100,
    sigma_inicial: float = None,
    tol: float = 1e-8,
    verbose: bool = False
):
    fn_name = funcao.__name__
    min_limit = globals()[f"{fn_name.upper()}_MIN_LIMIT"]
    max_limit = globals()[f"{fn_name.upper()}_MAX_LIMIT"]
    
    pop = inicializar_populacao(funcao, pop_size, d)
    fitness = np.array([funcao(ind) for ind in pop])
    sigma = calcular_sigma_inicial(funcao, sigma_inicial)
    
    best_X, best_fitness = pop[np.argmin(fitness)].copy(), np.min(fitness)
    
    historico_fitness = [best_fitness]
    historico_sigma = [sigma]
    
    for t in range(max_iter):
        elite_size = max(5, pop_size//3)
        elite = selecionar_elite(pop, fitness, elite_size)
        
        new_pop = []
        for _ in range(pop_size):
            parents = elite[np.random.choice(elite_size, 2, replace=False)]
            child = recombinar(parents)
            child = mutar(child, sigma, min_limit, max_limit)
            new_pop.append(child)
        
        pop = np.array(new_pop)
        fitness = np.array([funcao(ind) for ind in pop])
        
        current_best_idx = np.argmin(fitness)
        best_X, best_fitness = atualizar_melhor_solucao(pop, fitness, best_X, best_fitness)
        sigma = ajustar_sigma(sigma, fitness[current_best_idx] < best_fitness)
        sigma = max(sigma * 0.995, 1e-12)
        
        historico_fitness.append(best_fitness)
        historico_sigma.append(sigma)
        
        if verbose and t % 50 == 0:
            print(f"Iteração {t}: Fitness = {best_fitness:.6f}, Sigma = {sigma:.6f}")
        
        if best_fitness <= tol:
            break
            
    return best_X, best_fitness, historico_fitness, historico_sigma

def avaliar_resultados(solucao, fitness, fn_name):
    global_min = globals()[f"{fn_name.upper()}_GLOBAL_MINIMUM_PARAM"]
    distancias = np.abs(solucao - global_min)
    indices_ordenados = np.argsort(distancias)
    
    print(f"\n--- {fn_name.capitalize()} ---")
    print("5 variáveis mais próximas do ótimo:")
    for i in indices_ordenados[:5]:
        print(f"x_{i+1}: {solucao[i]:.8f} (Erro: {distancias[i]:.2e})")
    print(f"Erro global: {fitness - global_min:.2e}")

configs = {
    'ackley': {'max_iter': 300, 'pop_size': 40},
    'rastrigin': {'max_iter': 400, 'pop_size': 50},
    'schwefel': {'max_iter': 800, 'pop_size': 60, 'sigma_inicial': 200},
    'rosenbrock': {'max_iter': 1000, 'pop_size': 50, 'sigma_inicial': 0.5}
}

def plotar_convergencia(historico_fitness, fn_name):
    plt.figure(figsize=(10, 6))
    plt.plot(historico_fitness, 'b-', linewidth=2)
    plt.title(f'Convergência da Função {fn_name.capitalize()}')
    plt.xlabel('Iteração')
    plt.ylabel('Melhor Fitness')
    plt.yscale('log') 
    plt.grid(True)
    plt.show()

def plotar_evolucao_sigma(historico_sigma, fn_name):
    plt.figure(figsize=(10, 6))
    plt.plot(historico_sigma, 'r-', linewidth=2)
    plt.title(f'Evolução do Sigma para {fn_name.capitalize()}')
    plt.xlabel('Iteração')
    plt.ylabel('Valor do Sigma')
    plt.grid(True)
    plt.show()

def plotar_distribuicao_solucao(solucao, fn_name):
    global_min = globals()[f"{fn_name.upper()}_GLOBAL_MINIMUM_PARAM"]
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(solucao, 'bo', alpha=0.7)
    plt.axhline(y=global_min, color='r', linestyle='--')
    plt.title(f'Valores das Variáveis\n{fn_name.capitalize()}')
    plt.xlabel('Índice da Variável')
    plt.ylabel('Valor')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    distancias = np.abs(solucao - global_min)
    plt.bar(range(len(distancias)), distancias, alpha=0.7)
    plt.title('Distância ao Ótimo Global')
    plt.xlabel('Índice da Variável')
    plt.ylabel('Distância')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    funcoes = [ackley, rastrigin, schwefel, rosenbrock]
    d = 30
    
    for funcao in funcoes:
        fn_name = funcao.__name__
        cfg = configs[fn_name]
        
        print(f"\n=== Executando {fn_name.capitalize()} ===")
        solucao, fitness, historico_fitness, historico_sigma = estrategia_evolutiva_modular(
            funcao,
            d=d,
            max_iter=cfg.get('max_iter', 500),
            pop_size=cfg.get('pop_size', 50),
            sigma_inicial=cfg.get('sigma_inicial'),
            tol=1e-8,
            verbose=True
        )
        
        avaliar_resultados(solucao, fitness, fn_name)
        
        plotar_convergencia(historico_fitness, fn_name)
        plotar_evolucao_sigma(historico_sigma, fn_name)
        plotar_distribuicao_solucao(solucao, fn_name)