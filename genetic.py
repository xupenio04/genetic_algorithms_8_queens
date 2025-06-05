from gene import Gene, Individual, Populacao
import matplotlib.pyplot as plt

populations = []
for i in range(30):
    pop = Populacao(100)
    goat, tentativa = pop.solve()
    print(f"Resolução: {goat} | Tentativa: {tentativa}")
    populations.append(pop)
    
plt.plot(populations[0].all_mean)
plt.show()