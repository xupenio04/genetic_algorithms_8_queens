from gene import Gene, Individual, Populacao
import matplotlib.pyplot as plt
import numpy as np

populations = []
list_attempts= []
max_attempts = 10000
for i in range(30):
    pop = Populacao(50)
    goat, tentativa = pop.solve()
    list_attempts.append(tentativa)
    print(f"Resolução: {goat} | Tentativa: {tentativa} | Execução: {i+1}")
    populations.append(pop)
    
not_solved=[i for i in list_attempts if i==max_attempts]
solved=[i for i in list_attempts if i!=max_attempts]


print(f"Porcentagem de execuções resolvidas: {(len(solved)/len(list_attempts))*100:.2f}%")

mean_val = np.mean(solved)
std_val = np.std(solved)

print(f"Média das Tentativas: {mean_val:.2f}")
print(f"Desvio Padrão das Tentativas: {std_val:.2f}")

plt.plot(solved, label='Tentativas')
plt.axhline(mean_val, color='green', linestyle='--', label='Média das Tentativas')
plt.axhline(mean_val + std_val, color='purple', linestyle=':', label='Média + Desvio Padrão')
plt.axhline(mean_val - std_val, color='purple', linestyle=':', label='Média - Desvio Padrão')
plt.legend()
plt.xlabel("Execução")
plt.ylabel("Geração")
plt.title("Tentativas por Execução com Média e Desvio Padrão")
plt.grid(True)
plt.show()

ind_conv= [len(pop.max_fit_ind()) for pop in populations]
plt.plot(ind_conv, label='Indivíduos Convergentes')
plt.axhline(np.mean(ind_conv), color='green', linestyle='--', label='Média de Indivíduos Convergentes')
plt.legend()
plt.xlabel("Execução")
plt.ylabel("Número de Indivíduos Convergentes")
plt.title("Número de Indivíduos Convergentes por Execução")
plt.grid(True)
plt.show()

fit_mean = [pop.fitness_media() for pop in populations]

print(f"Média do fitness médio: {np.mean(fit_mean):.2f}")
print(f"Desvio Padrão do fitness médio: {np.std(fit_mean):.2f}")

plt.plot(fit_mean, label='Média de Fitness')
plt.axhline(np.mean(fit_mean), color='green', linestyle='--', label='Média de Fitness')
plt.axhline(np.mean(fit_mean) + np.std(fit_mean), color='purple', linestyle=':', label='Média + Desvio Padrão')
plt.axhline(np.mean(fit_mean) - np.std(fit_mean), color='purple', linestyle=':', label='Média - Desvio Padrão')
plt.legend()
plt.xlabel("Execução")
plt.ylabel("Média de Fitness")
plt.title("Média de Fitness por Execução")
plt.grid(True)
plt.show()


plt.plot(populations[0].all_mean, label='Média da População')
plt.plot(populations[0].all_best_ind, label='Melhor Indivíduo')
plt.legend()
plt.xlabel("Geração")
plt.ylabel("Fitness")
plt.title("Evolução da População")
plt.grid(True)
plt.show()


new_pop= Populacao(50)
goat, tentativa = new_pop.solve_all()
print(f" Tentativa: {tentativa}")

plt.plot(new_pop.all_mean, label='Média da População')
plt.plot(new_pop.all_best_ind, label='Melhor Indivíduo')
plt.legend()
plt.xlabel("Geração")
plt.ylabel("Fitness")
plt.title("Evolução da População")
plt.grid(True)
plt.show()




