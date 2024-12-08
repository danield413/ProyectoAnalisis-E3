import copy
import numpy as np
import time
from itertools import combinations

from data.cargarData import obtenerInformacionCSV
from utilidades.background import aplicarCondicionesBackground
from utilidades.marginalizacionInicial import aplicarMarginalizacion
from utilidades.utils import generarMatrizPresenteInicial, particionComplemento
from utilidades.utils import generarMatrizFuturoInicial
from utilidades.utils import elementosNoSistemaCandidato
from utilidades.utils import producto_tensorial
from utilidades.partirRepresentacion import partirRepresentacion
from utilidades.vectorProbabilidad import obtenerVectorProbabilidad
from utilidades.utils import calcularEMD
from utilidades.comparaciones import compararParticion

#? ----------------- ENTRADAS DE DATOS ---------------------------------

# from data.matrices import TPM
from data.matrices import subconjuntoSistemaCandidato
from data.matrices import subconjuntoElementos
from data.matrices import estadoActualElementos
_, _, TPM = obtenerInformacionCSV('csv/red5.csv')


#? ----------------- MATRIZ PRESENTE Y MATRIZ FUTURO ---------------------------------

matrizPresente = generarMatrizPresenteInicial( len(estadoActualElementos) )
# print(len(matrizPresente))
matrizFuturo = generarMatrizFuturoInicial(matrizPresente)

# print("matrizPresente", matrizPresente[0])
# print("matrizFuturo", matrizFuturo)
# print("filas matriz presente", len(matrizPresente))
# print("filas matriz futuro", len(matrizFuturo))



#? ----------------- APLICAR CONDICIONES DE BACKGROUND ---------------------------------

#? Elementos que no hacen parte del sistema cantidato
elementosBackground = elementosNoSistemaCandidato(estadoActualElementos, subconjuntoElementos)

#? Realizar una copia de las matrices para no modificar las originales
nuevaTPM = np.copy(TPM)
nuevaMatrizPresente = np.copy(matrizPresente)
nuevaMatrizFuturo = np.copy(matrizFuturo)


#? Ejecución de las condiciones de background
nuevaMatrizPresente, nuevaMatrizFuturo, nuevaTPM = aplicarCondicionesBackground(matrizPresente, nuevaTPM, elementosBackground, nuevaMatrizFuturo, estadoActualElementos)

# print("nuevaMatrizPresente", nuevaMatrizPresente, len(nuevaMatrizPresente))
# print("nuevaMatrizFuturo", nuevaMatrizFuturo, len(nuevaMatrizFuturo))
# print("nuevaTPM", nuevaTPM, len(nuevaTPM))

#? ----------------- APLICAR MARGINALIZACIÓN INICIAL ---------------------------------

nuevaMatrizPresente, nuevaMatrizFuturo, nuevaTPM, nuevosIndicesElementos = aplicarMarginalizacion(nuevaMatrizFuturo, nuevaTPM, elementosBackground, estadoActualElementos, nuevaMatrizPresente)

# print("nuevaMatrizPresente", nuevaMatrizPresente)
# print("nuevaMatrizFuturo", nuevaMatrizFuturo)
# print("nuevaTPM", nuevaTPM)

#?  ------------------------ DIVIDIR EN LA REPRESENTACION -----------------------------------
#? P(ABC t | ABC t+1) = P(ABC t | A t+1) X P(ABC t | B t+1) X P(ABC t | C t+1)

#* tomar el subconjunto de elementos (los de t y t+1) con su indice
elementosT = [elem for elem in subconjuntoSistemaCandidato if 't' in elem and 't+1' not in elem]
elementosT1 = [elem for elem in subconjuntoSistemaCandidato if 't+1' in elem]


indicesElementosT = {list(elem.keys())[0]: idx for idx, elem in enumerate(estadoActualElementos) if list(elem.keys())[0] in elementosT}

# print("elementosT1", elementosT1)
# print("indicesElementosT viejos y nuevos", indicesElementosT,  nuevosIndicesElementos)

#? Ejecución de la representación
partirMatricesPresentes, partirMatricesFuturas, partirMatricesTPM = partirRepresentacion(nuevaMatrizPresente, nuevaMatrizFuturo, nuevaTPM, elementosT1, nuevosIndicesElementos)


#* --------------------------------------------------------------------------------------------
#* --------------------------------------------------------------------------------------------
#* --------------------------------------------------------------------------------------------



def busqueda_local(nuevaTPM, subconjuntoElementos, subconjuntoSistemaCandidato, estadoActualElementos, maxIteraciones = 10):
    

    particion_actual, particion_actual_complemento = generarParticionInicial(subconjuntoSistemaCandidato)
    
    
    
    vectorParticionActual1 = obtenerVectorProbabilidad(
        particion_actual, partirMatricesPresentes, partirMatricesFuturas, partirMatricesTPM,
        estadoActualElementos, subconjuntoElementos, indicesElementosT,
        nuevaMatrizPresente, nuevaMatrizFuturo, nuevaTPM, elementosT
    )

    vectorParticionActual2 = obtenerVectorProbabilidad(
        particion_actual_complemento, partirMatricesPresentes, partirMatricesFuturas, partirMatricesTPM,
        estadoActualElementos, subconjuntoElementos, indicesElementosT,
        nuevaMatrizPresente, nuevaMatrizFuturo, nuevaTPM, elementosT
    )
    
    
    vectorFinal = producto_tensorial(vectorParticionActual1, vectorParticionActual2)
    menorValorEMD = compararParticion(vectorFinal, nuevaMatrizPresente, nuevaTPM, subconjuntoElementos, estadoActualElementos)
    
    print("PARTICION INICIAL ELEGIDA")
    print(particion_actual, particion_actual_complemento, "emd", menorValorEMD)
    
    print("EJECUCION DE LAS ITERACIONES \n")
    
    iteracion = 0
    while iteracion < maxIteraciones:
        
        print("\n iteracion: ", iteracion , "\n")
        print("Mejor partición actual", particion_actual, particion_actual_complemento, "emd", menorValorEMD, "\n")
        
        vecindario = generarVecindario(particion_actual, particion_actual_complemento)
        
        # for vecino in vecindario:
        #     print(vecino)
        hayMejoria = False
        for vecino in vecindario:
                particion1 = vecino[0]
                particion2 = vecino[1]
                
                vectorParticion1 = obtenerVectorProbabilidad(
                    particion1, partirMatricesPresentes, partirMatricesFuturas, partirMatricesTPM,
                    estadoActualElementos, subconjuntoElementos, indicesElementosT,
                    nuevaMatrizPresente, nuevaMatrizFuturo, nuevaTPM, elementosT
                )

                vectorParticion2 = obtenerVectorProbabilidad(
                    particion2, partirMatricesPresentes, partirMatricesFuturas, partirMatricesTPM,
                    estadoActualElementos, subconjuntoElementos, indicesElementosT,
                    nuevaMatrizPresente, nuevaMatrizFuturo, nuevaTPM, elementosT
                )
                
                vectorFinal = producto_tensorial(vectorParticion1, vectorParticion2)
                
                valorEMD = compararParticion(vectorFinal, nuevaMatrizPresente, nuevaTPM, subconjuntoElementos, estadoActualElementos)
                
                print("VECINA", particion1, particion2)
                
                if valorEMD < menorValorEMD:
                    hayMejoria = True
                    menorValorEMD = valorEMD
                    particion_actual = particion1
                    particion_actual_complemento = particion2
            
        if not hayMejoria:
            print("\n No se encontró mejora \n")
            print("Mejor partición encontrada", particion_actual, particion_actual_complemento, "emd", menorValorEMD, "\n")
            break
        iteracion += 1
    
    return particion_actual, particion_actual_complemento, menorValorEMD
            

def generarVecindario(particion_actual, particion_complemento, limite=10):
    vecindario = []

    # Desempaquetar las particiones
    actual_t, actual_t_mas_1 = particion_actual
    complemento_t, complemento_t_mas_1 = particion_complemento

    # Movimientos de nodos entre particiones (de complemento a actual)
    for k in range(1, len(complemento_t) + 1):
        for grupo in combinations(complemento_t, k):
            nueva_actual = [actual_t + list(grupo), list(actual_t_mas_1)]
            nuevo_complemento = [list(complemento_t), list(complemento_t_mas_1)]
            for nodo in grupo:
                nuevo_complemento[0].remove(nodo)

            # Evitar particiones vacías
            if nueva_actual[0] and nuevo_complemento[0]:
                vecindario.append((nueva_actual, nuevo_complemento))

    # Movimientos de nodos entre particiones (de actual a complemento)
    for k in range(1, len(actual_t) + 1):
        for grupo in combinations(actual_t, k):
            nueva_actual = [list(actual_t), list(actual_t_mas_1)]
            nuevo_complemento = [complemento_t + list(grupo), list(complemento_t_mas_1)]
            for nodo in grupo:
                nueva_actual[0].remove(nodo)

            # Evitar particiones vacías
            if nueva_actual[0] and nuevo_complemento[0]:
                vecindario.append((nueva_actual, nuevo_complemento))

    # Intercambios de nodos entre particiones
    for nodo_actual in actual_t:
        for nodo_complemento in complemento_t:
            nueva_actual = [list(actual_t), list(actual_t_mas_1)]
            nuevo_complemento = [list(complemento_t), list(complemento_t_mas_1)]

            # Realizar el intercambio
            nueva_actual[0].remove(nodo_actual)
            nueva_actual[0].append(nodo_complemento)
            nuevo_complemento[0].remove(nodo_complemento)
            nuevo_complemento[0].append(nodo_actual)

            # Evitar particiones vacías
            if nueva_actual[0] and nuevo_complemento[0]:
                vecindario.append((nueva_actual, nuevo_complemento))

    # Redistribuciones aleatorias (siempre asegurando que ambas particiones tengan elementos)
    todos_los_nodos_t = actual_t + complemento_t
    todos_los_nodos_t_mas_1 = actual_t_mas_1 + complemento_t_mas_1
    for _ in range(3):  # Generar algunas redistribuciones aleatorias
        np.random.shuffle(todos_los_nodos_t)
        np.random.shuffle(todos_los_nodos_t_mas_1)

        # Dividir los nodos en dos particiones
        corte_t = np.random.randint(1, len(todos_los_nodos_t))
        corte_t_mas_1 = np.random.randint(1, len(todos_los_nodos_t_mas_1))

        nueva_actual = [todos_los_nodos_t[:corte_t], todos_los_nodos_t_mas_1[:corte_t_mas_1]]
        nuevo_complemento = [todos_los_nodos_t[corte_t:], todos_los_nodos_t_mas_1[corte_t_mas_1:]]

        # Asegurarse de que ambas particiones no estén vacías
        if nueva_actual[0] and nuevo_complemento[0]:
            vecindario.append((nueva_actual, nuevo_complemento))

    # Mezclar el vecindario y limitar su tamaño
    np.random.shuffle(vecindario)
    return vecindario[:limite]
    


def generarParticionInicial(subconjuntoSistemaCandidato):
    # Número de elementos para dividir
    num_elementos = len(subconjuntoSistemaCandidato)

    # Elegir un tamaño aleatorio para la partición inicial
    tamano_particion = np.random.randint(1, num_elementos)

    # Generar todas las combinaciones posibles de ese tamaño
    combinaciones = list(combinations(subconjuntoSistemaCandidato, tamano_particion))

    # Elegir una combinación al azar para formar la partición inicial
    particion_aleatoria = list(combinaciones[np.random.randint(0, len(combinaciones))])

    # Crear la partición inicial y su complemento
    particionInicial = ([], [])
    for elemento in particion_aleatoria:
        if 't+1' in elemento:
            particionInicial[0].append(elemento)
        else:
            particionInicial[1].append(elemento)

    complemento = particionComplemento(particionInicial, subconjuntoSistemaCandidato)

    return particionInicial, complemento

busqueda_local(nuevaTPM, subconjuntoElementos, subconjuntoSistemaCandidato, estadoActualElementos)

    
    
    