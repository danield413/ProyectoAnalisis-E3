from itertools import combinations
import numpy as np

def generarVecindario(particion_actual, particion_complemento, limite=10):
    vecindario = []

    # Desempaquetar las particiones
    actual_t, actual_t_mas_1 = particion_actual
    complemento_t, complemento_t_mas_1 = particion_complemento

    # Generar movimientos de uno o más nodos de complemento a actual
    for k in range(1, len(complemento_t) + 1):  # Variar tamaño del grupo movido
        for grupo in combinations(complemento_t, k):
            nueva_actual = [actual_t + list(grupo), list(actual_t_mas_1)]
            nuevo_complemento = [list(complemento_t), list(complemento_t_mas_1)]
            for nodo in grupo:
                nuevo_complemento[0].remove(nodo)

            # Considerar particiones válidas con elementos en una sublista
            if nueva_actual[0] or nueva_actual[1]:
                vecindario.append((nueva_actual, nuevo_complemento))

    # Generar movimientos de uno o más nodos de actual a complemento
    for k in range(1, len(actual_t) + 1):  # Variar tamaño del grupo movido
        for grupo in combinations(actual_t, k):
            nueva_actual = [list(actual_t), list(actual_t_mas_1)]
            nuevo_complemento = [complemento_t + list(grupo), list(complemento_t_mas_1)]
            for nodo in grupo:
                nueva_actual[0].remove(nodo)

            # Considerar particiones válidas con elementos en una sublista
            if nueva_actual[0] or nueva_actual[1]:
                vecindario.append((nueva_actual, nuevo_complemento))

    # Opcional: Generar particiones con sublistas vacías controladas
    # Mover todos los elementos de actual_t a complemento_t y viceversa
    if actual_t:
        nueva_actual = [[], list(actual_t_mas_1)]
        nuevo_complemento = [complemento_t + list(actual_t), list(complemento_t_mas_1)]
        vecindario.append((nueva_actual, nuevo_complemento))

    if complemento_t:
        nueva_actual = [list(actual_t) + list(complemento_t), list(actual_t_mas_1)]
        nuevo_complemento = [[], list(complemento_t_mas_1)]
        vecindario.append((nueva_actual, nuevo_complemento))

    # Intercambio de nodos entre particiones
    for nodo_actual in actual_t:
        for nodo_complemento in complemento_t:
            nueva_actual = [list(actual_t), list(actual_t_mas_1)]
            nuevo_complemento = [list(complemento_t), list(complemento_t_mas_1)]

            # Intercambiar los nodos
            nueva_actual[0].remove(nodo_actual)
            nueva_actual[0].append(nodo_complemento)
            nuevo_complemento[0].remove(nodo_complemento)
            nuevo_complemento[0].append(nodo_actual)

            vecindario.append((nueva_actual, nuevo_complemento))

    # Mezclar aleatoriamente y limitar tamaño
    np.random.shuffle(vecindario)
    return vecindario[:limite]


particion_actual = (['bt+1'], [])
particion_complemento = (['at+1', 'ct+1', 'dt+1'], ['at', 'bt', 'ct', 'dt'])

vecindario = generarVecindario(particion_actual, particion_complemento, limite=10)

for vecino in vecindario:
    print(vecino)
    
    