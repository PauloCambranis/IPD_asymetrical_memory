# Simulacion del Dilema del Prisionero Iterado con Algoritmos Geneticos
# VERSION BALANCEADA: Mantiene representacion igual de cada tipo de memoria

import numpy as np
import csv
import os
from datetime import datetime
from organism_serializer import OrganismSerializer
# %% # Clase Decision: Red neuronal que decide cooperar o desertar

class Decision:
    def __init__(self, tipo_jugador, capas_ocultas=[]): # Constructor con arquitectura dinamica
        """
        Inicializa una red neuronal con arquitectura configurable.

        Parametros:
        - tipo_jugador: int, profundidad de memoria (2, 3, 4, etc.)
        - capas_ocultas: lista de ints, numero de neuronas en cada capa oculta
                        Ejemplos: [] = sin capas (red simple)
                                  [4] = una capa oculta con 4 neuronas
                                  [8, 4] = dos capas ocultas con 8 y 4 neuronas
        """
        self.tipo_jugador = tipo_jugador
        self.capas_ocultas = capas_ocultas # Guarda la arquitectura de capas ocultas
        self.entradas = self.tipo_jugador * 2 # Las entradas = 2 * memoria (propias + oponente)

        # Listas para almacenar pesos y sesgos de todas las capas
        self.pesos = [] # Lista de matrices de pesos entre capas
        self.sesgos = [] # Lista de vectores de sesgos para cada capa

        # Construye la arquitectura capa por capa
        tamanhos_capas = [self.entradas] + capas_ocultas + [1] # [entrada, ocultas..., salida]

        for i in range(len(tamanhos_capas) - 1):
            # Crea matriz de pesos entre capa i y capa i+1
            peso = np.random.randn(tamanhos_capas[i], tamanhos_capas[i+1]) * 0.5
            sesgo = np.random.randn(tamanhos_capas[i+1]) * 0.5
            self.pesos.append(peso)
            self.sesgos.append(sesgo)

    def sigmoid(self, x): # Funcion de activacion sigmoide
        return 1 / (1 + np.exp(-np.clip(x, -500, 500))) # Calcula sigmoide con limites

    def decidir(self, historial): # Metodo que decide la accion basandose en el historial
        entrada = np.array(historial).flatten() # Convierte el historial en un vector unidimensional

        # Propagacion hacia adelante a traves de todas las capas
        activacion = entrada
        for i in range(len(self.pesos)):
            # Calcula z = activacion * pesos + sesgo
            z = np.dot(activacion, self.pesos[i]) + self.sesgos[i]
            # Aplica funcion de activacion sigmoide
            activacion = self.sigmoid(z)

        # La ultima activacion es la salida
        salida = activacion
        return 1 if salida[0] > 0.5 else -1 # Devuelve 1 (cooperar) o -1
# %% # Clase Jugador: Agente que juega el IPD

class Jugador:
    def __init__(self, tipo_jugador, capas_ocultas=[]): # Constructor con arquitectura dinamica
        """
        Crea un jugador con una red neuronal de arquitectura configurable.

        Parametros:
        - tipo_jugador: int, profundidad de memoria (2, 3, 4, etc.)
        - capas_ocultas: lista de ints, arquitectura de la red neuronal
                        Ejemplos: [] = red simple sin capas ocultas
                                  [4] = una capa oculta con 4 neuronas
                                  [8, 6, 4] = tres capas ocultas
        """
        self.tipo_jugador = tipo_jugador # Guarda la profundidad de memoria
        self.capas_ocultas = capas_ocultas # Guarda la arquitectura de la red
        self.decision = Decision(tipo_jugador, capas_ocultas) # Crea la red neuronal
        self.puntuacion = 0 # Inicializa la puntuacion en 0


    def jugar(self, historial): # Metodo que ejecuta una movida
        # Si no hay suficiente historial, rellena con 0s
        if len(historial[0]) < self.tipo_jugador:
            # Crea vectores rellenados con 0s al inicio
            hist_propio = [0] * (self.tipo_jugador - len(historial[0])) + historial[0]
            hist_oponente = [0] * (self.tipo_jugador - len(historial[1])) + historial[1]
            historial_reciente = [hist_propio, hist_oponente]
        else:
            # Usa solo los ultimos N movimientos segun la memoria
            historial_reciente = [
                historial[0][-self.tipo_jugador:], # ultimos N movimientos propios
                historial[1][-self.tipo_jugador:] # ultimos N movimientos del oponente
            ]
        return self.decision.decidir(historial_reciente) # Usa la red neuronal para decidir

    def reproducir(self, desviacion_std=0.3): # Metodo que crea un hijo clonando al padre con ruido gaussiano
        """
        Crea un hijo clonando al padre y agregando ruido gaussiano a todos los pesos y sesgos.
        Funciona con cualquier arquitectura de red neuronal.
        """
        hijo = Jugador(self.tipo_jugador, self.capas_ocultas) # Crea un nuevo jugador del mismo tipo

        # Copia todos los pesos y sesgos de todas las capas agregando ruido gaussiano
        for i in range(len(self.decision.pesos)):
            # Copia la matriz de pesos de la capa i y agrega ruido
            hijo.decision.pesos[i] = self.decision.pesos[i].copy() + np.random.randn(*self.decision.pesos[i].shape) * desviacion_std
            # Copia el vector de sesgos de la capa i y agrega ruido
            hijo.decision.sesgos[i] = self.decision.sesgos[i].copy() + np.random.randn(*self.decision.sesgos[i].shape) * desviacion_std

        return hijo # Devuelve el jugador hijo creado
# %% # Clase Juego: Orquesta los partidos del Dilema del Prisionero
class Juego:
    def jugar_ronda(self, jugador1, jugador2, historial1, historial2): # Ejecuta una ronda entre dos jugadores
        movimiento1 = jugador1.jugar(historial1) # Jugador 1 decide su movimiento
        movimiento2 = jugador2.jugar(historial2) # Jugador 2 decide su movimiento

        if movimiento1 == 1 and movimiento2 == 1: # Si ambos cooperan (1, 1)
            puntos1, puntos2 = 3, 3 # Ambos reciben 3 puntos
        elif movimiento1 == -1 and movimiento2 == 1: # Si jugador 1 deserta (-1) y 2 coopera (1)
            puntos1, puntos2 = 5, 0 # Jugador 1 recibe 5, jugador 2 recibe 0
        elif movimiento1 == 1 and movimiento2 == -1: # Si jugador 1 coopera (1) y 2 deserta (-1)
            puntos1, puntos2 = 0, 5 # Jugador 1 recibe 0, jugador 2 recibe 5
        else: # Si ambos desertan (-1, -1)
            puntos1, puntos2 = 1, 1 # Ambos reciben 1 punto

        return movimiento1, movimiento2, puntos1, puntos2 # Devuelve movimientos y puntos

    def jugar_partido(self, jugador1, jugador2, rondas=300): # Juega un partido completo
        historial1 = [[], []] # Historial para jugador 1: [sus movimientos, movimientos del oponente]
        historial2 = [[], []] # Historial para jugador 2: [sus movimientos, movimientos del oponente]
        puntos_totales1 = 0 # Acumulador de puntos para jugador 1
        puntos_totales2 = 0 # Acumulador de puntos para jugador 2

        for ronda in range(rondas): # Repite por cada ronda del partido
            mov1, mov2, p1, p2 = self.jugar_ronda(jugador1, jugador2, historial1, historial2) # Ejecuta una ronda
            historial1[0].append(mov1) # Agrega el movimiento de jugador 1 a su historial
            historial1[1].append(mov2) # Agrega el movimiento del oponente al historial de jugador 1
            historial2[0].append(mov2) # Agrega el movimiento de jugador 2 a su historial
            historial2[1].append(mov1) # Agrega el movimiento del oponente al historial de jugador 2
            puntos_totales1 += p1 # Suma los puntos de la ronda al total de jugador 1
            puntos_totales2 += p2 # Suma los puntos de la ronda al total de jugador 2

        return puntos_totales1, puntos_totales2 # Devuelve los puntos totales de ambos

    def jugar_torneo(self, jugadores): # Ejecuta un torneo round-robin con todos los jugadores
        puntuaciones = [0] * len(jugadores) # Lista para almacenar puntuaciones de cada jugador

        for i in range(len(jugadores)): # Recorre cada jugador como jugador 1
            for j in range(len(jugadores)): # Recorre cada jugador como jugador 2
                puntos1, puntos2 = self.jugar_partido(jugadores[i], jugadores[j]) # Juega partido entre i y j
                puntuaciones[i] += puntos1 # Suma puntos al jugador i

        return puntuaciones # Devuelve la lista de puntuaciones finales
# %% Clase Grabador: Registra estadisticas de la simulacion
class Grabador:
    def __init__(self): # Constructor que inicializa las estructuras de datos
        self.datos = [] # Lista para almacenar datos de cada generacion

    def grabar_generacion(self, generacion, jugadores, puntuaciones): # Registra datos de una generacion
        for i in range(len(jugadores)): # Recorre cada jugador
            self.datos.append({ # Agrega un diccionario con los datos del jugador
                'generacion': generacion, # Numero de generacion
                'jugador': i, # indice del jugador
                'tipo': jugadores[i].tipo_jugador, # Tipo de memoria del jugador
                'arquitectura': str(jugadores[i].capas_ocultas), # Arquitectura de la red neuronal
                'puntuacion': puntuaciones[i] # Puntuacion obtenida
            })

    def guardar_csv(self, nombre_archivo): # Guarda los datos en un archivo CSV
        with open(nombre_archivo, 'w', newline='') as archivo: # Abre archivo en modo escritura
            campos = ['generacion', 'jugador', 'tipo', 'arquitectura', 'puntuacion'] # Define las columnas
            escritor = csv.DictWriter(archivo, fieldnames=campos) # Crea escritor CSV
            escritor.writeheader() # Escribe la fila de encabezados
            escritor.writerows(self.datos) # Escribe todas las filas de datos

    def obtener_datos(self): # Retorna los datos registrados
        return self.datos

    def obtener_estadisticas(self): # Calcula estadisticas resumidas
        if not self.datos: # Si no hay datos registrados
            return {} # Devuelve diccionario vacio

        generaciones = {} # Diccionario para agrupar por generacion
        for dato in self.datos: # Recorre cada registro
            gen = dato['generacion'] # Extrae el numero de generacion
            if gen not in generaciones: # Si la generacion no esta en el diccionario
                generaciones[gen] = [] # Crea una lista para esa generacion
            generaciones[gen].append(dato['puntuacion']) # Agrega la puntuacion

        estadisticas = {} # Diccionario para las estadisticas calculadas
        for gen, puntos in generaciones.items(): # Recorre cada generacion
            estadisticas[gen] = { # Calcula estadisticas para la generacion
                'promedio': np.mean(puntos), # Puntuacion promedio
                'maximo': np.max(puntos), # Puntuacion maxima
                'minimo': np.min(puntos) # Puntuacion minima
            }

        return estadisticas # Devuelve el diccionario de estadisticas
# %% Clase Evolucion: Gestiona el proceso evolutivo CON BALANCEO DE TIPOS
class Evolucion:
    def __init__(self): # Constructor que inicializa la poblacion
        self.poblacion = [] # Lista de jugadores en la poblacion
        self.tipos_jugador = [] # Lista de tipos de jugador (memoria, arquitectura)
        self.contador_generacion = 0 # Contador de generaciones transcurridas
        self.juego = Juego() # Instancia del juego para ejecutar torneos

    def inicializar_poblacion(self, n_jugadores, tipos_jugador): # Crea la poblacion inicial
        """
        Crea la poblacion inicial con diferentes tipos de jugadores.
        Distribuye EQUITATIVAMENTE los jugadores entre los tipos.

        Parametros:
        - n_jugadores: int, numero total de jugadores a crear
        - tipos_jugador: lista de tuplas (memoria, arquitectura)
                         Ejemplos: (2, []) = memoria 2, sin capas ocultas
                                   (3, [4]) = memoria 3, una capa oculta de 4 neuronas
                                   (4, [8, 4]) = memoria 4, dos capas ocultas
        """
        self.poblacion = [] # Limpia la poblacion
        self.tipos_jugador = tipos_jugador # Guarda los tipos de jugador

        # Calcula cuantos jugadores por tipo
        jugadores_por_tipo = n_jugadores // len(tipos_jugador)
        residuo = n_jugadores % len(tipos_jugador)

        jugadores_creados = 0
        for tipo_idx, tipo in enumerate(tipos_jugador):
            # El residuo se distribuye entre los primeros tipos
            cantidad = jugadores_por_tipo + (1 if tipo_idx < residuo else 0)

            for _ in range(cantidad):
                memoria = tipo[0]
                arquitectura = tipo[1] if len(tipo) > 1 else []
                jugador = Jugador(memoria, arquitectura)
                self.poblacion.append(jugador)
                jugadores_creados += 1

    def evaluar_aptitud(self, grabador): # Ejecuta torneo y registra resultados
        puntuaciones = self.juego.jugar_torneo(self.poblacion) # Ejecuta el torneo
        grabador.grabar_generacion(self.contador_generacion, self.poblacion, puntuaciones) # Registra datos
        return puntuaciones # Devuelve las puntuaciones

    def seleccion_balanceada(self, puntuaciones): # Selecciona los mejores jugadores POR TIPO
        """
        Selecciona los mejores jugadores mantiendo representacion igual de cada tipo.

        Estrategia:
        1. Agrupa jugadores por tipo de memoria
        2. Selecciona los mejores de cada grupo mantiendo proporcion igual
        3. Garantiza que ningun tipo se extinga
        """
        # Agrupar jugadores por tipo
        jugadores_por_tipo = {}
        for idx, jugador in enumerate(self.poblacion):
            tipo_clave = (jugador.tipo_jugador, tuple(jugador.capas_ocultas))
            if tipo_clave not in jugadores_por_tipo:
                jugadores_por_tipo[tipo_clave] = []
            jugadores_por_tipo[tipo_clave].append((idx, puntuaciones[idx]))

        # Seleccionar los mejores de cada tipo (mitad de 50 = 25 por tipo, distribuidos equitativamente)
        nuevos_mejores = []
        mejores_por_tipo = max(1, 50 // len(jugadores_por_tipo))  # Al menos 1 por tipo

        for tipo_clave, jugadores_del_tipo in jugadores_por_tipo.items():
            # Ordena por puntuacion descendente
            jugadores_del_tipo.sort(key=lambda x: x[1], reverse=True)
            # Toma los mejores de este tipo
            mejores = jugadores_del_tipo[:mejores_por_tipo]
            nuevos_mejores.extend(mejores)

        # Si no tenemos exactamente 50, ajustamos tomando los mejores globales
        if len(nuevos_mejores) < 50:
            todos_jugadores = [(idx, puntuaciones[idx]) for idx in range(len(self.poblacion))]
            todos_jugadores.sort(key=lambda x: x[1], reverse=True)
            nuevos_mejores = todos_jugadores[:50]
        elif len(nuevos_mejores) > 50:
            # Si tenemos mas de 50, tomamos solo los 50 mejores
            nuevos_mejores.sort(key=lambda x: x[1], reverse=True)
            nuevos_mejores = nuevos_mejores[:50]

        indices = [idx for idx, _ in nuevos_mejores]
        self.poblacion = [self.poblacion[i] for i in indices]

    def reproducir_poblacion_balanceada(self, desviacion_std=0.3): # Crea nueva generacion MANTENIENDO BALANCE
        """
        Crea la nueva generacion reproduciendo en proporcion igual de cada tipo.
        Garantiza que cada tipo tenga exactamente 50 descendientes.
        """
        nueva_poblacion = []

        # Agrupar poblacion seleccionada por tipo
        jugadores_por_tipo = {}
        for jugador in self.poblacion:
            tipo_clave = (jugador.tipo_jugador, tuple(jugador.capas_ocultas))
            if tipo_clave not in jugadores_por_tipo:
                jugadores_por_tipo[tipo_clave] = []
            jugadores_por_tipo[tipo_clave].append(jugador)

        # Crear descendientes proporcionalmente de cada tipo
        descendientes_por_tipo = 50 // len(jugadores_por_tipo)
        residuo = 50 % len(jugadores_por_tipo)

        tipo_idx = 0
        for tipo_clave, padres in jugadores_por_tipo.items():
            cantidad_descendientes = descendientes_por_tipo + (1 if tipo_idx < residuo else 0)

            for _ in range(cantidad_descendientes):
                padre = padres[_ % len(padres)]  # Cicla entre los padres disponibles
                hijo = padre.reproducir(desviacion_std)
                nueva_poblacion.append(hijo)

            tipo_idx += 1

        self.poblacion.extend(nueva_poblacion) # Agrega los nuevos individuos a la poblacion

    def ejecutar_generacion(self, grabador): # Ejecuta un ciclo completo de generacion
        puntuaciones = self.evaluar_aptitud(grabador) # Evalua aptitud mediante torneo
        self.seleccion_balanceada(puntuaciones) # Selecciona los mejores manteniendo balance
        self.reproducir_poblacion_balanceada() # Crea nueva generacion manteniendo balance
        self.contador_generacion += 1 # Incrementa el contador de generaciones

    def guardar_poblacion(self, serializer, experiment_path): # Guarda la poblacion actual
        """Guarda los pesos de toda la poblacion actual"""
        for idx, jugador in enumerate(self.poblacion):
            serializer.save_organism(jugador, experiment_path, self.contador_generacion, idx)

# %% MAIN MAIN MIAN MAIN MAINMAIN

if __name__ == "__main__": # Si se ejecuta como programa principal
    print("=" * 80)
    print("SIMULACION DEL DILEMA DEL PRISIONERO ITERADO (VERSION BALANCEADA)")
    print("=" * 80)
    print("Esta version mantiene representacion igual de cada tipo de memoria\n")

    # Crear carpeta de experimento
    base_folder = "/Users/pauloc/Masters/1 Sem/IA/Proyecto/experiments"
    serializer = OrganismSerializer(base_folder)
    experiment_path = serializer.create_experiment_folder("ipd_simulation_balanced")

    print(f"Experimento guardado en: {experiment_path}\n")

    evolucion = Evolucion() # Crea instancia del gestor evolutivo
    grabador = Grabador() # Crea instancia del grabador de datos

    # Define los tipos de jugadores con arquitecturas configurables:
    # Formato: (memoria, [capas_ocultas])
    tipos = [
        (2, []),      # Memoria 2, sin capas ocultas (red simple)
        #(3, [6]),      # Memoria 3, 6 capas ocultas (Fogel simple)
        (3, [20]),     # Memoria 3, una capa oculta con 20 neuronas (Fogel original)
        #(4, [6]),      # Memoria 4,
        #(4, [20]),     # Memoria 5,
        (5, [20, 4]),  # Memoria 4, dos capas ocultas con 8 y 4 neuronas (profunda)
    ]

    evolucion.inicializar_poblacion(100, tipos) # Crea poblacion inicial de 100 individuos

    gens = 200
    for gen in range(gens):
        print(f"Generacion {gen+1}/{gens}...") # Imprime el numero de generacion actual
        evolucion.ejecutar_generacion(grabador) # Ejecuta una generacion completa
        evolucion.guardar_poblacion(serializer, experiment_path) # Guarda pesos de la poblacion

    estadisticas = grabador.obtener_estadisticas() # Obtiene estadisticas de todas las generaciones
    print("\n" + "=" * 80)
    print("ESTADISTICAS FINALES")
    print("=" * 80)
    for gen, stats in estadisticas.items(): # Recorre cada generacion
        print(f"Gen {gen}: Promedio={stats['promedio']:.1f}, Max={stats['maximo']}, Min={stats['minimo']}") # Imprime estadisticas

    # Guardar CSV en carpeta de experimento
    csv_results_path = os.path.join(experiment_path, "results", "resultados.csv")
    serializer.save_csv(None, grabador.obtener_datos(), experiment_path)

    print("\n" + "=" * 80)
    print("✓ Simulacion completada")
    print(f"✓ Resultados guardados en: {experiment_path}")
    print(f"✓ CSV: {csv_results_path}")
    print(f"✓ Organisms: {os.path.join(experiment_path, 'organisms/')}")
    print("=" * 80)
