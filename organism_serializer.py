# Utilidades para serializar y deserializar pesos de organismos
# Soporta preservación de pesos de redes neuronales para auditoría externa

import pickle
import os
import json
from datetime import datetime

class OrganismSerializer:
    """
    Gestiona la serialización y deserialización de pesos de organismos.
    Almacena organismos en formato pickle con un registro CSV para búsqueda fácil.
    """

    def __init__(self, base_folder):
        """
        Inicializa el serializador con una carpeta base de salida.

        Parámetros:
        - base_folder: str, ruta de la carpeta donde se almacenan los experimentos
        """
        self.base_folder = base_folder
        os.makedirs(base_folder, exist_ok=True)

    def create_experiment_folder(self, experiment_name):
        """
        Crea una estructura de carpetas para un experimento específico.

        Parámetros:
        - experiment_name: str, nombre del experimento (ej: 'duelo_memoria2_vs_memoria4')

        Devuelve:
        - experiment_path: str, ruta de la carpeta del experimento
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_folder = f"{experiment_name}_{timestamp}"
        experiment_path = os.path.join(self.base_folder, experiment_folder)

        # Crear subdirectorios
        os.makedirs(os.path.join(experiment_path, "organisms"), exist_ok=True)
        os.makedirs(os.path.join(experiment_path, "results"), exist_ok=True)
        os.makedirs(os.path.join(experiment_path, "analysis"), exist_ok=True)

        return experiment_path

    def save_organism(self, organism, experiment_path, generacion, jugador_id, tipo_poblacion=None):
        """
        Guarda los pesos de un organismo en un archivo pickle.

        Parámetros:
        - organism: objeto Jugador con decisión (red neuronal)
        - experiment_path: str, ruta de la carpeta del experimento
        - generacion: int, número de generación
        - jugador_id: int, ID del jugador
        - tipo_poblacion: str, tipo de población opcional ('A', 'B', etc.)

        Devuelve:
        - filename: str, ruta del archivo guardado
        """
        organisms_folder = os.path.join(experiment_path, "organisms")

        # Crear nombre de archivo
        if tipo_poblacion:
            filename = f"gen{generacion:04d}_player{jugador_id:04d}_type{tipo_poblacion}.pkl"
        else:
            filename = f"gen{generacion:04d}_player{jugador_id:04d}.pkl"

        filepath = os.path.join(organisms_folder, filename)

        # Preparar datos para guardar
        organism_data = {
            'tipo_jugador': organism.tipo_jugador,
            'capas_ocultas': organism.capas_ocultas,
            'tipo_poblacion': tipo_poblacion,
            'pesos': organism.decision.pesos,
            'sesgos': organism.decision.sesgos,
            'generacion': generacion,
            'jugador_id': jugador_id,
        }

        # Guardar usando pickle
        with open(filepath, 'wb') as f:
            pickle.dump(organism_data, f)

        return filepath

    def load_organism(self, filepath):
        """
        Carga un organismo desde un archivo pickle.

        Parámetros:
        - filepath: str, ruta del archivo pickle

        Devuelve:
        - organism_data: dict con toda la información del organismo
        """
        with open(filepath, 'rb') as f:
            organism_data = pickle.load(f)

        return organism_data

    def create_registry(self, experiment_path, data_dict):
        """
        Crea un registro de todos los organismos guardados.

        Parámetros:
        - experiment_path: str, ruta de la carpeta del experimento
        - data_dict: dict con claves como 'tipo_poblacion', 'generacion', etc.
        """
        registry_path = os.path.join(experiment_path, "results", "organism_registry.json")

        with open(registry_path, 'w') as f:
            json.dump(data_dict, f, indent=2)

    def save_csv(self, csv_path, data, experiment_path):
        """
        Mueve resultados CSV a la carpeta de resultados del experimento.

        Parámetros:
        - csv_path: str, ruta original del CSV
        - data: list de dicts (los registros)
        - experiment_path: str, ruta de la carpeta del experimento
        """
        import csv

        results_path = os.path.join(experiment_path, "results", "resultados.csv")

        if not data:
            return results_path

        # Obtener nombres de campos del primer registro
        fieldnames = list(data[0].keys())

        with open(results_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

        return results_path


class OrganismReconstructior:
    """
    Reconstruye organismos desde pesos guardados para auditoría externa.
    """

    @staticmethod
    def reconstruct_from_pickle(pickle_path):
        """
        Reconstruye un organismo completo desde un archivo pickle.

        Parámetros:
        - pickle_path: str, ruta del archivo pickle

        Devuelve:
        - organism: organismo reconstruido listo para pruebas
        """
        # Importar aquí para evitar importaciones circulares
        import sys
        sys.path.insert(0, os.path.dirname(__file__))

        # Determinar qué módulo importar basado en el contenido del pickle
        with open(pickle_path, 'rb') as f:
            organism_data = pickle.load(f)

        # Determinar módulo basado en si existe tipo_poblacion
        if organism_data.get('tipo_poblacion'):
            # Este es un organismo de duelo
            from ipd_duel import Jugador, Decision
        else:
            # Este es estándar o balanceado - intentar ambos
            try:
                from ipd_simulation import Jugador, Decision
            except ImportError:
                from ipd_simulation_balanced import Jugador, Decision

        # Reconstruir el organismo
        # Verificar si es un organismo de duelo (tiene tipo_poblacion)
        if organism_data.get('tipo_poblacion'):
            # Versión duelo - acepta parámetro tipo_poblacion
            organism = Jugador(organism_data['tipo_jugador'],
                              organism_data['capas_ocultas'],
                              tipo_poblacion=organism_data.get('tipo_poblacion'))
        else:
            # Versión estándar/balanceada - sin parámetro tipo_poblacion
            organism = Jugador(organism_data['tipo_jugador'],
                              organism_data['capas_ocultas'])

        # Restaurar pesos y sesgos
        organism.decision.pesos = organism_data['pesos']
        organism.decision.sesgos = organism_data['sesgos']

        return organism

    @staticmethod
    def get_organism_info(pickle_path):
        """
        Obtiene información sobre un organismo sin reconstruirlo.

        Parámetros:
        - pickle_path: str, ruta del archivo pickle

        Devuelve:
        - info: dict con metadatos del organismo
        """
        with open(pickle_path, 'rb') as f:
            organism_data = pickle.load(f)

        return {
            'tipo_jugador': organism_data['tipo_jugador'],
            'capas_ocultas': organism_data['capas_ocultas'],
            'tipo_poblacion': organism_data.get('tipo_poblacion'),
            'generacion': organism_data.get('generacion'),
            'jugador_id': organism_data.get('jugador_id'),
            'num_weights': sum(len(p.flatten()) for p in organism_data['pesos']),
            'num_biases': sum(len(b) for b in organism_data['sesgos']),
        }
