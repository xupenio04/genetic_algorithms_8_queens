import random

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
    def __init__(self, num_genes:int, vector_genes:Gene = None):
        if vector_genes is None:
            individual_genes = list(range(num_genes))
            random.shuffle(individual_genes)
            self.genes = [Gene(num) for num in individual_genes]
        else:
            self.genes = vector_genes 
    
    def __str__(self):
        return f'{[gene.get_number10() for gene in self.genes]}'
    
    def __repr__(self):
        return f'{[gene.get_number10() for gene in self.genes]}'

    def print_genes_bits(self):
        print(self.genes)