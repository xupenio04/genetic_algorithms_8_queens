from gene import Individual, Gene

genes_quantity = 8
population_size = 20
population = []

for i in range(population_size):
    population.append(Individual(genes_quantity))

for individual in population:
    individual.print_genes_bits()
