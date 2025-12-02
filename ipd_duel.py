# Simulacion del Dilema del Prisionero Iterado: Duelo entre dos tipos de jugadores
# Dos arquitecturas diferentes compiten exclusivamente entre si
# Poblaciones balanceadas: la mejor mitad de cada tipo surviva

import csv
import os
from datetime import datetime

import numpy as np

from organism_serializer import OrganismSerializer

# %% # Clase Decision: Red neuronal que decide cooperar o desertar


class Decision:
    def __init__(
        self, tipo_jugador, capas_ocultas=[]
    ):  # Constructor con arquitectura dinamica
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
        self.capas_ocultas = capas_ocultas  # Guarda la arquitectura de capas ocultas
        self.entradas = (
            self.tipo_jugador * 2
        )  # Las entradas = 2 * memoria (propias + oponente)

        # Listas para almacenar pesos y sesgos de todas las capas
        self.pesos = []  # Lista de matrices de pesos entre capas
        self.sesgos = []  # Lista de vectores de sesgos para cada capa

        # Construye la arquitectura capa por capa
        tamanhos_capas = (
            [self.entradas] + capas_ocultas + [1]
        )  # [entrada, ocultas..., salida]

        for i in range(len(tamanhos_capas) - 1):
            # Crea matriz de pesos entre capa i y capa i+1
            peso = np.random.randn(tamanhos_capas[i], tamanhos_capas[i + 1]) * 0.5
            sesgo = np.random.randn(tamanhos_capas[i + 1]) * 0.5
            self.pesos.append(peso)
            self.sesgos.append(sesgo)

    def sigmoid(self, x):  # Funcion de activacion sigmoide
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Calcula sigmoide con limites

    def decidir(
        self, historial
    ):  # Metodo que decide la accion basandose en el historial
        entrada = np.array(
            historial
        ).flatten()  # Convierte el historial en un vector unidimensional

        # Propagacion hacia adelante a traves de todas las capas
        activacion = entrada
        for i in range(len(self.pesos)):
            # Calcula z = activacion * pesos + sesgo
            z = np.dot(activacion, self.pesos[i]) + self.sesgos[i]
            # Aplica funcion de activacion sigmoide
            activacion = self.sigmoid(z)

        # La ultima activacion es la salida
        salida = activacion
        return 1 if salida[0] > 0.5 else -1  # Devuelve 1 (cooperar) o -1


# %% # Clase Jugador: Agente que juega el IPD


class Jugador:
    def __init__(
        self, tipo_jugador, capas_ocultas=[], tipo_poblacion="A"
    ):  # Constructor con arquitectura dinamica
        """
        Crea un jugador con una red neuronal de arquitectura configurable.

        Parametros:
        - tipo_jugador: int, profundidad de memoria (2, 3, 4, etc.)
        - capas_ocultas: lista de ints, arquitectura de la red neuronal
        - tipo_poblacion: str, 'A' o 'B' para identificar la poblacion
        """
        self.tipo_jugador = tipo_jugador  # Guarda la profundidad de memoria
        self.capas_ocultas = capas_ocultas  # Guarda la arquitectura de la red
        self.decision = Decision(tipo_jugador, capas_ocultas)  # Crea la red neuronal
        self.puntuacion = 0  # Inicializa la puntuacion en 0
        self.tipo_poblacion = tipo_poblacion  # Identifica si es tipo A o B

    def jugar(self, historial):  # Metodo que ejecuta una movida
        # Si no hay suficiente historial, rellena con 0s
        if len(historial[0]) < self.tipo_jugador:
            # Crea vectores rellenados con 0s al inicio
            hist_propio = [0] * (self.tipo_jugador - len(historial[0])) + historial[0]
            hist_oponente = [0] * (self.tipo_jugador - len(historial[1])) + historial[1]
            historial_reciente = [hist_propio, hist_oponente]
        else:
            # Usa solo los ultimos N movimientos segun la memoria
            historial_reciente = [
                historial[0][-self.tipo_jugador :],  # ultimos N movimientos propios
                historial[1][
                    -self.tipo_jugador :
                ],  # ultimos N movimientos del oponente
            ]
        return self.decision.decidir(
            historial_reciente
        )  # Usa la red neuronal para decidir

    def reproducir(
        self, desviacion_std=0.3
    ):  # Metodo que crea un hijo clonando al padre con ruido gaussiano
        """
        Crea un hijo clonando al padre y agregando ruido gaussiano a todos los pesos y sesgos.
        """
        hijo = Jugador(self.tipo_jugador, self.capas_ocultas, self.tipo_poblacion)

        # Copia todos los pesos y sesgos de todas las capas agregando ruido gaussiano
        for i in range(len(self.decision.pesos)):
            # Copia la matriz de pesos de la capa i y agrega ruido
            hijo.decision.pesos[i] = (
                self.decision.pesos[i].copy()
                + np.random.randn(*self.decision.pesos[i].shape) * desviacion_std
            )
            # Copia el vector de sesgos de la capa i y agrega ruido
            hijo.decision.sesgos[i] = (
                self.decision.sesgos[i].copy()
                + np.random.randn(*self.decision.sesgos[i].shape) * desviacion_std
            )

        return hijo  # Devuelve el jugador hijo creado


# %% # Clase Juego: Orquesta los partidos del Dilema del Prisionero


class Juego:
    def jugar_ronda(
        self, jugador1, jugador2, historial1, historial2
    ):  # Ejecuta una ronda entre dos jugadores
        movimiento1 = jugador1.jugar(historial1)  # Jugador 1 decide su movimiento
        movimiento2 = jugador2.jugar(historial2)  # Jugador 2 decide su movimiento

        if movimiento1 == 1 and movimiento2 == 1:  # Si ambos cooperan (1, 1)
            puntos1, puntos2 = 3, 3  # Ambos reciben 3 puntos
        elif (
            movimiento1 == -1 and movimiento2 == 1
        ):  # Si jugador 1 deserta (-1) y 2 coopera (1)
            puntos1, puntos2 = 5, 0  # Jugador 1 recibe 5, jugador 2 recibe 0
        elif (
            movimiento1 == 1 and movimiento2 == -1
        ):  # Si jugador 1 coopera (1) y 2 deserta (-1)
            puntos1, puntos2 = 0, 5  # Jugador 1 recibe 0, jugador 2 recibe 5
        else:  # Si ambos desertan (-1, -1)
            puntos1, puntos2 = 1, 1  # Ambos reciben 1 punto

        return (
            movimiento1,
            movimiento2,
            puntos1,
            puntos2,
        )  # Devuelve movimientos y puntos

    def jugar_partido(
        self, jugador1, jugador2, rondas=300
    ):  # Juega un partido completo
        historial1 = [
            [],
            [],
        ]  # Historial para jugador 1: [sus movimientos, movimientos del oponente]
        historial2 = [
            [],
            [],
        ]  # Historial para jugador 2: [sus movimientos, movimientos del oponente]
        puntos_totales1 = 0  # Acumulador de puntos para jugador 1
        puntos_totales2 = 0  # Acumulador de puntos para jugador 2

        for ronda in range(rondas):  # Repite por cada ronda del partido
            mov1, mov2, p1, p2 = self.jugar_ronda(
                jugador1, jugador2, historial1, historial2
            )  # Ejecuta una ronda
            historial1[0].append(
                mov1
            )  # Agrega el movimiento de jugador 1 a su historial
            historial1[1].append(
                mov2
            )  # Agrega el movimiento del oponente al historial de jugador 1
            historial2[0].append(
                mov2
            )  # Agrega el movimiento de jugador 2 a su historial
            historial2[1].append(
                mov1
            )  # Agrega el movimiento del oponente al historial de jugador 2
            puntos_totales1 += p1  # Suma los puntos de la ronda al total de jugador 1
            puntos_totales2 += p2  # Suma los puntos de la ronda al total de jugador 2

        return puntos_totales1, puntos_totales2  # Devuelve los puntos totales de ambos

    def jugar_torneo_duel(
        self, jugadores_a, jugadores_b
    ):  # Torneo donde A solo juega contra B
        """
        Ejecuta un torneo donde jugadores tipo A juegan SOLO contra jugadores tipo B.
        Devuelve las puntuaciones de ambas poblaciones.
        """
        puntuaciones_a = [0] * len(jugadores_a)  # Puntuaciones para tipo A
        puntuaciones_b = [0] * len(jugadores_b)  # Puntuaciones para tipo B

        # Cada jugador A juega contra cada jugador B
        for i in range(len(jugadores_a)):  # Recorre cada jugador A
            for j in range(len(jugadores_b)):  # Recorre cada jugador B
                puntos_a, puntos_b = self.jugar_partido(
                    jugadores_a[i], jugadores_b[j]
                )  # Juega partido
                puntuaciones_a[i] += puntos_a  # Suma puntos al jugador A
                puntuaciones_b[j] += puntos_b  # Suma puntos al jugador B

        return (
            puntuaciones_a,
            puntuaciones_b,
        )  # Devuelve puntuaciones de ambas poblaciones


# %% Clase Grabador: Registra estadisticas de la simulacion


class Grabador:
    def __init__(self):  # Constructor que inicializa las estructuras de datos
        self.datos = []  # Lista para almacenar datos de cada generacion

    def grabar_generacion(
        self, generacion, jugadores_a, jugadores_b, puntuaciones_a, puntuaciones_b
    ):  # Registra datos de una generacion
        """
        Registra datos de ambas poblaciones en una generacion.
        """
        # Registra jugadores tipo A
        for i in range(len(jugadores_a)):
            self.datos.append(
                {
                    "generacion": generacion,
                    "jugador": i,
                    "tipo_poblacion": "A",
                    "tipo_memoria": jugadores_a[i].tipo_jugador,
                    "arquitectura": str(jugadores_a[i].capas_ocultas),
                    "puntuacion": puntuaciones_a[i],
                }
            )

        # Registra jugadores tipo B
        for i in range(len(jugadores_b)):
            self.datos.append(
                {
                    "generacion": generacion,
                    "jugador": i,
                    "tipo_poblacion": "B",
                    "tipo_memoria": jugadores_b[i].tipo_jugador,
                    "arquitectura": str(jugadores_b[i].capas_ocultas),
                    "puntuacion": puntuaciones_b[i],
                }
            )

    def guardar_csv(self, nombre_archivo):  # Guarda los datos en un archivo CSV
        with open(
            nombre_archivo, "w", newline=""
        ) as archivo:  # Abre archivo en modo escritura
            campos = [
                "generacion",
                "jugador",
                "tipo_poblacion",
                "tipo_memoria",
                "arquitectura",
                "puntuacion",
            ]  # Define las columnas
            escritor = csv.DictWriter(archivo, fieldnames=campos)  # Crea escritor CSV
            escritor.writeheader()  # Escribe la fila de encabezados
            escritor.writerows(self.datos)  # Escribe todas las filas de datos

    def obtener_datos(self):  # Retorna los datos registrados
        return self.datos

    def obtener_estadisticas(self):  # Calcula estadisticas resumidas
        if not self.datos:  # Si no hay datos registrados
            return {}  # Devuelve diccionario vacio

        generaciones = {}  # Diccionario para agrupar por generacion
        for dato in self.datos:  # Recorre cada registro
            gen = dato["generacion"]  # Extrae el numero de generacion
            if gen not in generaciones:  # Si la generacion no esta en el diccionario
                generaciones[gen] = {"A": [], "B": []}  # Crea listas para ambos tipos
            generaciones[gen][dato["tipo_poblacion"]].append(
                dato["puntuacion"]
            )  # Agrega la puntuacion

        estadisticas = {}  # Diccionario para las estadisticas calculadas
        for gen, datos_por_tipo in generaciones.items():  # Recorre cada generacion
            estadisticas[gen] = {
                "promedio_A": np.mean(datos_por_tipo["A"])
                if datos_por_tipo["A"]
                else 0,
                "promedio_B": np.mean(datos_por_tipo["B"])
                if datos_por_tipo["B"]
                else 0,
                "maximo_A": np.max(datos_por_tipo["A"]) if datos_por_tipo["A"] else 0,
                "maximo_B": np.max(datos_por_tipo["B"]) if datos_por_tipo["B"] else 0,
                "minimo_A": np.min(datos_por_tipo["A"]) if datos_por_tipo["A"] else 0,
                "minimo_B": np.min(datos_por_tipo["B"]) if datos_por_tipo["B"] else 0,
            }

        return estadisticas  # Devuelve el diccionario de estadisticas


# %% Clase Evolucion: Gestiona el proceso evolutivo CON DOS POBLACIONES BALANCEADAS


class Evolucion:
    def __init__(self):  # Constructor que inicializa las poblaciones
        self.poblacion_a = []  # Poblacion tipo A
        self.poblacion_b = []  # Poblacion tipo B
        self.contador_generacion = 0  # Contador de generaciones transcurridas
        self.juego = Juego()  # Instancia del juego para ejecutar torneos

    def inicializar_poblacion(
        self, n_jugadores, tipo_a, tipo_b
    ):  # Crea las poblaciones iniciales
        """
        Crea dos poblaciones iniciales de jugadores con diferentes arquitecturas.

        Parametros:
        - n_jugadores: int, numero total de jugadores por poblacion (e.g., 50 tipo A + 50 tipo B = 100 total)
        - tipo_a: tupla (memoria, [capas_ocultas]) para poblacion A
        - tipo_b: tupla (memoria, [capas_ocultas]) para poblacion B
        """
        self.poblacion_a = []  # Limpia poblacion A
        self.poblacion_b = []  # Limpia poblacion B

        memoria_a, arquitectura_a = tipo_a[0], tipo_a[1] if len(tipo_a) > 1 else []
        memoria_b, arquitectura_b = tipo_b[0], tipo_b[1] if len(tipo_b) > 1 else []

        # Crea jugadores tipo A
        for _ in range(n_jugadores):
            jugador = Jugador(memoria_a, arquitectura_a, tipo_poblacion="A")
            self.poblacion_a.append(jugador)

        # Crea jugadores tipo B
        for _ in range(n_jugadores):
            jugador = Jugador(memoria_b, arquitectura_b, tipo_poblacion="B")
            self.poblacion_b.append(jugador)

        print(
            f"Poblaciones inicializadas: {len(self.poblacion_a)} tipo A, {len(self.poblacion_b)} tipo B"
        )

    def evaluar_aptitud(self, grabador):  # Ejecuta torneo y registra resultados
        puntuaciones_a, puntuaciones_b = self.juego.jugar_torneo_duel(
            self.poblacion_a, self.poblacion_b
        )
        grabador.grabar_generacion(
            self.contador_generacion,
            self.poblacion_a,
            self.poblacion_b,
            puntuaciones_a,
            puntuaciones_b,
        )
        return puntuaciones_a, puntuaciones_b

    def seleccion_balanceada(
        self, poblacion, puntuaciones
    ):  # Selecciona la mejor mitad de una poblacion
        """
        Selecciona exactamente la mitad superior (los mejores) de una poblacion.
        Garantiza que la poblacion nunca se extinga.
        """
        indices = np.argsort(puntuaciones)[
            ::-1
        ]  # Ordena indices por puntuacion descendente
        top_n = len(poblacion) // 2  # Toma exactamente la mitad
        indices_seleccionados = indices[:top_n]
        return [poblacion[i] for i in indices_seleccionados]

    def reproducir_poblacion(
        self, poblacion_seleccionada, desviacion_std=0.3
    ):  # Crea nueva generacion
        """
        Reproduce la poblacion seleccionada para volver al tamaño original.
        Cada individuo seleccionado genera exactamente 2 descendientes.
        """
        nueva_poblacion = []
        tamaño_original = len(poblacion_seleccionada) * 2  # Vuelve al tamaño original

        for i in range(tamaño_original):
            padre = poblacion_seleccionada[
                i % len(poblacion_seleccionada)
            ]  # Cicla entre padres
            hijo = padre.reproducir(desviacion_std)
            nueva_poblacion.append(hijo)

        return nueva_poblacion

    def ejecutar_generacion(self, grabador):  # Ejecuta un ciclo completo de generacion
        """
        Ejecuta una generacion completa:
        1. Evalua aptitud (torneo)
        2. Selecciona la mejor mitad de cada poblacion
        3. Reproduce para volver al tamaño original
        """
        puntuaciones_a, puntuaciones_b = self.evaluar_aptitud(
            grabador
        )  # Evalua aptitud mediante torneo

        # Selecciona la mejor mitad de cada poblacion
        self.poblacion_a = self.seleccion_balanceada(self.poblacion_a, puntuaciones_a)
        self.poblacion_b = self.seleccion_balanceada(self.poblacion_b, puntuaciones_b)

        # Reproduce para restaurar tamaño original
        self.poblacion_a = self.reproducir_poblacion(self.poblacion_a)
        self.poblacion_b = self.reproducir_poblacion(self.poblacion_b)

        self.contador_generacion += 1  # Incrementa el contador de generaciones

    def guardar_poblaciones(
        self, serializer, experiment_path
    ):  # Guarda ambas poblaciones
        """Guarda los pesos de ambas poblaciones actuales"""
        for idx, jugador in enumerate(self.poblacion_a):
            serializer.save_organism(
                jugador,
                experiment_path,
                self.contador_generacion,
                idx,
                tipo_poblacion="A",
            )
        for idx, jugador in enumerate(self.poblacion_b):
            serializer.save_organism(
                jugador,
                experiment_path,
                self.contador_generacion,
                idx,
                tipo_poblacion="B",
            )


# %% MAIN MAIN MAIN MAIN MAIN

if __name__ == "__main__":  # Si se ejecuta como programa principal
    print("=" * 80)
    print("SIMULACION DEL DILEMA DEL PRISIONERO ITERADO: DUELO DE DOS POBLACIONES")
    print("=" * 80)
    print("Dos tipos de jugadores con arquitecturas diferentes compiten entre si.")
    print("Cada poblacion: mejor 50% sobrevive y crea descendientes\n")

    # Crear carpeta de experimento
    base_folder = "/Users/pauloc/Masters/1 Sem/IA/Proyecto/experiments"
    serializer = OrganismSerializer(base_folder)
    experiment_path = serializer.create_experiment_folder("ipd_duel")

    print(f"Experimento guardado en: {experiment_path}\n")

    evolucion = Evolucion()  # Crea instancia del gestor evolutivo
    grabador = Grabador()  # Crea instancia del grabador de datos

    # Define los dos tipos de jugadores (A y B)
    # Formato: (memoria, [capas_ocultas])
    tipo_a = (2, [8])  # Poblacion A: Memoria 2, sin capas ocultas (simple)
    tipo_b = (
        6,
        [20, 4],
    )  # Poblacion B: Memoria 4, una capa oculta con 20 neuronas (compleja)

    print(
        f"Tipo A: Memoria {tipo_a[0]}, Arquitectura {tipo_a[1] if tipo_a[1] else 'simple'}"
    )
    print(
        f"Tipo B: Memoria {tipo_b[0]}, Arquitectura {tipo_b[1] if tipo_b[1] else 'simple'}\n"
    )

    evolucion.inicializar_poblacion(
        50, tipo_a, tipo_b
    )  # Crea 50 jugadores de cada tipo

    gens = 500
    for gen in range(gens):
        print(f"Generacion {gen + 1}/{gens}...")
        evolucion.ejecutar_generacion(grabador)
        evolucion.guardar_poblaciones(
            serializer, experiment_path
        )  # Guarda pesos de ambas poblaciones

    estadisticas = (
        grabador.obtener_estadisticas()
    )  # Obtiene estadisticas de todas las generaciones
    print("\n" + "=" * 80)
    print("ESTADISTICAS FINALES")
    print("=" * 80)
    for gen, stats in estadisticas.items():
        print(
            f"Gen {gen:3d}: A={stats['promedio_A']:7.1f} (max={stats['maximo_A']:7.0f}), "
            f"B={stats['promedio_B']:7.1f} (max={stats['maximo_B']:7.0f})"
        )

    # Guardar CSV en carpeta de experimento
    csv_results_path = os.path.join(experiment_path, "results", "resultados.csv")
    serializer.save_csv(None, grabador.obtener_datos(), experiment_path)

    print("\n" + "=" * 80)
    print("✓ Simulacion completada")
    print(f"✓ Resultados guardados en: {experiment_path}")
    print(f"✓ CSV: {csv_results_path}")
    print(f"✓ Organisms: {os.path.join(experiment_path, 'organisms/')}")
    print("=" * 80)
