import copy
from itertools import chain, combinations
import numpy as np
import time

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
_, _, TPM = obtenerInformacionCSV('csv/red6.csv')


#? ----------------- MATRIZ PRESENTE Y MATRIZ FUTURO ---------------------------------

matrizPresente = generarMatrizPresenteInicial( len(estadoActualElementos) )
print(len(matrizPresente))
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
# print("------ REPRESENTACIÓN -----------")
partirMatricesPresentes, partirMatricesFuturas, partirMatricesTPM = partirRepresentacion(nuevaMatrizPresente, nuevaMatrizFuturo, nuevaTPM, elementosT1, nuevosIndicesElementos)

print("partirMatricesPresentes", partirMatricesPresentes)
print("partirMatricesFuturas", partirMatricesFuturas)
print("partirMatricesTPM", partirMatricesTPM)

particionesCandidatas = []
listaDeUPrimas = []


# print("------ ALGORITMO -----------")
# def fuerzaBruta(nuevaTPM, subconjuntoElementos, subconjuntoSistemaCandidato, estadoActualElementos):
    
#     print(subconjuntoSistemaCandidato)

# start_time = time.time()
# x = fuerzaBruta(nuevaTPM, subconjuntoElementos, subconjuntoSistemaCandidato, estadoActualElementos)
# print("candidatas")
# end_time = time.time()

# Conjunto base
# subconjuntoSistemaCandidato = [
#     'at', 'bt', 'ct', 'dt', 'et', 'ft',
#     'at+1', 'bt+1', 'ct+1', 'dt+1', 'et+1', 'ft+1'
# ]

def fuerzabruta(subconjuntoSistemaCandidato):
    particiones = []

    # Función para obtener todas las combinaciones posibles de un conjunto
    def obtener_subconjuntos(conjunto):
        return chain.from_iterable(combinations(conjunto, r) for r in range(len(conjunto) + 1))
    
    # Separar elementos en t y t+1
    elementost = [elem for elem in subconjuntoSistemaCandidato if 't' in elem and 't+1' not in elem]
    elementost1 = [elem for elem in subconjuntoSistemaCandidato if 't+1' in elem]
    
    # Generar todas las particiones
    for subset_t in obtener_subconjuntos(elementost):
        for subset_t1 in obtener_subconjuntos(elementost1):
            particiones.append((list(subset_t1), list(subset_t)))
    
    
    particionesResultantes = []
    for x in particiones:
        if x[0] == [] and x[1] == []:
            continue
        print(x, particionComplemento(x, subconjuntoSistemaCandidato))
        #* calcular el emd
        particionNormal = x 
        complemento = particionComplemento(x, subconjuntoSistemaCandidato)
        
        vectorParticion1 = obtenerVectorProbabilidad(
            particionNormal, partirMatricesPresentes, partirMatricesFuturas, partirMatricesTPM,
            estadoActualElementos, subconjuntoElementos, indicesElementosT,
            nuevaMatrizPresente, nuevaMatrizFuturo, nuevaTPM, elementosT
        )
        
        vectorParticion2 = obtenerVectorProbabilidad(
            complemento, partirMatricesPresentes, partirMatricesFuturas, partirMatricesTPM,
            estadoActualElementos, subconjuntoElementos, indicesElementosT,
            nuevaMatrizPresente, nuevaMatrizFuturo, nuevaTPM, elementosT
        )
        
        vectorFinal = producto_tensorial(vectorParticion1, vectorParticion2)
        emd = compararParticion(vectorFinal, nuevaMatrizPresente, nuevaTPM, subconjuntoElementos, estadoActualElementos)
        
        particionesResultantes.append({
            'particion1': x,
            'particion2': complemento,
            'emd': emd
        })
        
    
    return particionesResultantes
   
def encontrar_minimo_emd(arreglo_diccionarios):
    # Encuentra el menor valor de 'emd' en el arreglo
    menor_emd = min(d['emd'] for d in arreglo_diccionarios)
    
    # Filtra los diccionarios que tienen el valor mínimo de 'emd'
    resultado = [d for d in arreglo_diccionarios if d['emd'] == menor_emd]
    
    return resultado


time_start = time.time()
particiones = fuerzabruta(subconjuntoSistemaCandidato)
time_end = time.time()

print("Tiempo de ejecución: ", time_end - time_start)

print("\n particiones")
for particion in encontrar_minimo_emd(particiones):
    print(particion)