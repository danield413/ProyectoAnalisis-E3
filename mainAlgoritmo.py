import copy
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
    # print("diferencia EMD", menorValorEMD)
    # print()
    
    
    iteracion = 0
    while iteracion < maxIteraciones:
        
        print()
        print("Particion Inicial al ejecutar el ciclo")
        print(particion_actual, particion_actual_complemento)
        print("emd" , menorValorEMD)
        print()
        
        vecindario = generarVecindario(particion_actual, particion_actual_complemento)
    
        for vecino in vecindario:
                print("particion vecina")
                particion1 = vecino[0]
                particion2 = vecino[1]
                print(particion1, particion2)
                
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
                
                print("vector particion 1", vectorParticion1)
                print("vector particion 2", vectorParticion2)
                
                vectorFinal = producto_tensorial(vectorParticion1, vectorParticion2)
                
                valorEMD = compararParticion(vectorFinal, nuevaMatrizPresente, nuevaTPM, subconjuntoElementos, estadoActualElementos)
                
                print("valor EMD", valorEMD)
                
                if valorEMD < menorValorEMD:
                    menorValorEMD = valorEMD
                    particion_actual = particion1
                    particion_actual_complemento = particion2
                    
                
                # print("vector_tpm", nuevaTPM)
                # diferenciaEMD = calcularEMD(vectorFinal, vectorParticion1)
                # print("diferencia EMD", diferenciaEMD , "\n")
                
                
                
            
        iteracion += 1
            
        
    

def generarVecindario(particion_actual, particion_actual_complemento):
    # Inicializamos una lista vacía para almacenar las particiones vecinas
    vecindario = []
    
    particionVencidarioActual = copy.deepcopy(particion_actual)
    particionVecindarioActualComplemento = copy.deepcopy(particion_actual_complemento)
    
    #identificar cual particion es mas grande
    for comp in particion_actual_complemento[0]:
        ##la agrego a la 1era particion
        particionVencidarioActual[0].append(comp)
        ##la elimino de la 2da particion
        particionVecindarioActualComplemento[0].remove(comp)
        # print("particion vecina")
        # print(particionVencidarioActual, particionVecindarioActualComplemento)
        vecindario.append((particionVencidarioActual, particionVecindarioActualComplemento))
        
    particionVencidarioActual = copy.deepcopy(particion_actual)
    particionVecindarioActualComplemento = copy.deepcopy(particion_actual_complemento)
    
    for comp in particion_actual_complemento[1]:
        ##la agrego a la 2da particion
        particionVencidarioActual[1].append(comp)
        ##la elimino de la 1era particion
        particionVecindarioActualComplemento[1].remove(comp)
        # print("particion vecina")
        # print(particionVencidarioActual, particionVecindarioActualComplemento)
        vecindario.append((particionVencidarioActual, particionVecindarioActualComplemento))
        
    return vecindario


        

def generarParticionInicial(suconjuntoSistemaCandidato):
    
    numeroRandom = np.random.randint(1, len(subconjuntoSistemaCandidato))
    elementoRandom = subconjuntoSistemaCandidato[numeroRandom]
    
    particionInicial = ([],[])
    if 't+1' in elementoRandom:
        particionInicial[0].append(elementoRandom)
    else:
        particionInicial[1].append(elementoRandom)
        
    complemento = particionComplemento(particionInicial, subconjuntoSistemaCandidato)
    
    return particionInicial, complemento

    
busqueda_local(nuevaTPM, subconjuntoElementos, subconjuntoSistemaCandidato, estadoActualElementos)
    
    
    
    