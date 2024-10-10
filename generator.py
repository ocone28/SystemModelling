import numpy as np
import matplotlib.pyplot as plt

# Funzione per generare variate casuali
def inverse_transform_sampling(n):
    random_numbers = []
    for _ in range(n):
        u = np.random.uniform(0, 1)  # Genera un numero casuale uniforme tra 0 e 1
        if u <= 0.25:
            # Se u è nell'intervallo [2, 3] cdf invertita
            x = 2 + 2 * np.sqrt(u)
        else:
            # Se u è nell'intervallo (3, 6] cdf invertita
            x = 6 - 3 * np.sqrt(2 * (1 - u))
        random_numbers.append(x)
    return np.array(random_numbers)

# Genera una sequenza di 1000 numeri
n_samples = 1000
samples = inverse_transform_sampling(n_samples)

# Costruisce un istogramma
plt.hist(samples, bins=30, density=True, edgecolor='black', alpha=0.7)
plt.title('Random Variates Histogram')
plt.xlabel('x')
plt.ylabel('Density')
plt.show()
