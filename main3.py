import numpy as np
from scipy.stats import norm

#case1
a_x0=8
a_a=13
a_m=16

#case2
b_x0=8
b_a=11
b_m=30

#case3
c_x0=7
c_a=7
c_m=16

#case4
d_x0=8
d_a=7
d_m=25

vector_case1=[]
vector_case2=[]
vector_case3=[]
vector_case4=[]

value1=a_x0
value2=b_x0
value3=c_x0
value4=d_x0


for i in range(30):
    #gererazione prossimo numero case 1
    case1_new_value=(a_a * value1) % a_m
    value1=case1_new_value
    vector_case1.append(case1_new_value/a_m)

    # gererazione prossimo numero case 2
    case2_new_value = (b_a * value2) % b_m
    value2 = case2_new_value
    vector_case2.append(case2_new_value/b_m)

    # gererazione prossimo numero case 3
    case3_new_value = (c_a * value3) % c_m
    value3 = case3_new_value
    vector_case3.append(case3_new_value/c_m)

    # gererazione prossimo numero case 4
    case4_new_value = (d_a * value4) % d_m
    value4 = case4_new_value
    vector_case4.append(case4_new_value/d_m)


print(vector_case1)
print(vector_case2)
print(vector_case3)
print(vector_case4)

# # normalizzo il vettore 1 in un range 0-1
# def normalize_vector(vector):
#
#     vector1 = np.array(vector)
#     # Calcolo del minimo e del massimo
#     min_val = np.min(vector1)
#     max_val = np.max(vector1)
#     # Verifica se il massimo e il minimo sono uguali (evita divisione per zero)
#     if max_val == min_val:
#         vettore_normalizzato = np.zeros(vector1.shape)  # Restituisce un vettore di zeri se tutti i valori sono uguali
#     else:
#         vettore_normalizzato = (vector1 - min_val) / (max_val - min_val)
#
#     return vettore_normalizzato
#
#
# n_vector1=normalize_vector(vector_case1)
# n_vector2=normalize_vector(vector_case2)
# n_vector3=normalize_vector(vector_case3)
# n_vector4=normalize_vector(vector_case4)


# Funzione per il test di Kolmogorov-Smirnov
def kolmogorov_smirnov_test(data):
    # Step 1: Rank the data from smallest to largest
    N = len(data)
    sorted_data = np.sort(data)

    # Step 2: Compute D+ and D-
    D_plus = np.max([(i+1)/N - sorted_data[i] for i in range(N)])
    D_minus = np.max([sorted_data[i] - i/N for i in range(N)])

    # Step 3: Compute D = max(D+, D-)
    D = max(D_plus, D_minus)

    # Step 4: Usa il valore critico dalla tabella per N = 30 e alpha = 0.05
    D_alpha = 1.36 / np.sqrt(N)  # D_alpha tabulato per N > 30 e alpha=0.05


    # Step 5: Confronta D con D_alpha
    if D > D_alpha:
        result = "Reject: Data do not come from a uniform distribution."
    else:
        result = "Success: Data are consistent with a uniform distribution."

    return D, D_alpha, result

# Esegui il test per ciascun vettore normalizzato
D1, D_alpha1, result1 = kolmogorov_smirnov_test(vector_case1)
D2, D_alpha2, result2 = kolmogorov_smirnov_test(vector_case2)
D3, D_alpha3, result3 = kolmogorov_smirnov_test(vector_case3)
D4, D_alpha4, result4 = kolmogorov_smirnov_test(vector_case4)

# Output dei risultati
print(f"Case 1: D = {"%.2f" %D1}, D_alpha = {"%.2f" %D_alpha1}, Result: {result1}")
print(f"Case 2: D = {"%.2f" %D2}, D_alpha = {"%.2f" %D_alpha2}, Result: {result2}")
print(f"Case 3: D = {"%.2f" %D3}, D_alpha = {"%.2f" %D_alpha3}, Result: {result3}")
print(f"Case 4: D = {"%.2f" %D4}, D_alpha = {"%.2f" %D_alpha4}, Result: {result4}")



def autocorrelation_schmidt_taylor(data, i=0, k=1, alpha=0.05):
    N = len(data)

    # Calcola il massimo numero di coppie che puoi usare garantendo che non ci siano indici fuori range
    M = (N - i - k - 1) // k  # Calcolo corretto di M

    if M < 1:
        print("Non ci sono abbastanza dati per calcolare l'autocorrelazione.")
        return

    # Calcola la stima dell'autocorrelazione
    autocorr_sum = sum(data[i + j * k] * data[i + (j + 1) * k] for j in range(M + 1))
    rho_hat = (autocorr_sum / (M + 1)) - 0.25

    # Calcola la deviazione standard
    sigma_rho_hat = np.sqrt((13 * M + 7) / (12 * (M + 1)))

    # Verifica che la deviazione standard non sia troppo piccola (per evitare Z_0 enormi)
    if sigma_rho_hat == 0:
        print("Deviazione standard troppo piccola, impossibile calcolare Z0.")
        return

    # Calcola la statistica Z0
    Z0 = rho_hat / sigma_rho_hat

    # Trova il valore critico z_alpha/2 dalla distribuzione normale
    z_alpha_2 = norm.ppf(1 - alpha / 2)

    # Verifica se si rifiuta l'ipotesi nulla
    if -z_alpha_2 <= Z0 <= z_alpha_2:
        print(f"Autocorrelazione stimata: {"%.2f" %rho_hat}, Z0: {"%.2f" %Z0}, Non si rifiuta l'ipotesi nulla di indipendenza.")
    else:
        print(
            f"Autocorrelazione stimata: {"%.2f" %rho_hat}, Z0: {"%.2f" %Z0}, Si rifiuta l'ipotesi nulla: autocorrelazione significativa.")




autocorrelation_schmidt_taylor(vector_case1, i=0, k=1, alpha=0.05)
autocorrelation_schmidt_taylor(vector_case2, i=0, k=1, alpha=0.05)
autocorrelation_schmidt_taylor(vector_case3, i=0, k=1, alpha=0.05)
autocorrelation_schmidt_taylor(vector_case4, i=0, k=1, alpha=0.05)


# print(f"Autocorrelazione per lag k=1: {autocorr1:.4f}")
# print(f"Autocorrelazione per lag k=1: {autocorr2:.4f}")
# print(f"Autocorrelazione per lag k=1: {autocorr3:.4f}")
# print(f"Autocorrelazione per lag k=1: {autocorr4:.4f}")
