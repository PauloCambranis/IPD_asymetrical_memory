# Simulación del Dilema del Prisionero Iterado con Algoritmos Genéticos

Una simulación de estrategias emergentes en el Dilema del Prisionero Iterado (IPD) usando redes neuronales evolucionadas con algoritmos genéticos. Explora cómo agentes con diferentes capacidades cognitivas (memoria e inteligencia) desarrollan estrategias competitivas.

## Descripción del Proyecto

Este proyecto estudia la evolución de estrategias en el IPD mediante:
- **Redes neuronales totalmente conectadas** como genoma evolucionable
- **Algoritmos genéticos** para selección y reproducción con mutación
- **Competencia de torneo round-robin** donde todos los jugadores compiten entre sí
- **Diferentes tipos de memoria y arquitecturas** para analizar trade-offs cognitivos

### Preguntas de Investigación

- ¿Proporciona mayor memoria ventajas adaptativas?
- ¿Emergen espontáneamente estrategias tipo "Tit-for-Tat"?
- ¿Existe una meseta de beneficios más allá de cierta capacidad de memoria?
- ¿Cómo afecta la arquitectura de la red al desempeño evolutivo?
- ¿Qué dinámicas poblacionales emergen con selección balanceada vs. libre?

## Instalación

### Requisitos

- Python 3.7+
- NumPy
- Pandas (opcional, para análisis posterior)

### Configuración

```bash
# Clonar el repositorio
git clone <repo-url>
cd Proyecto

# Instalar dependencias
pip install numpy pandas
```

## Estructura de Archivos

```
├── ipd_simulation.py                # Simulación estándar (competencia libre)
├── ipd_simulation_balanced.py       # Simulación balanceada (mantiene proporciones)
├── ipd_duel.py                      # Simulación de duelo (dos poblaciones enfrentadas)
├── organism_serializer.py           # Guardar/cargar pesos de organismos
├── audit_organisms.ipynb            # Notebook para analizar organismos evolucionados
└── README.md                         # Este archivo
```

## Cómo Ejecutar las Simulaciones

### 1. Simulación Estándar

La simulación estándar permite que todos los tipos de jugadores compitan libremente. Los tipos con mejor desempeño pueden dominar y potencialmente causar extinción de otros tipos.

```bash
python ipd_simulation.py
```

**Salida esperada:**
- Directorio de experimento con timestamp: `experiments/ipd_simulation_standard_YYYYMMDD_HHMMSS/`
- CSV con resultados: `results/resultados.csv`
- Archivos pickle con pesos: `organisms/gen####_player####.pkl`

### 2. Simulación Balanceada

Mantiene representación igual de cada tipo de memoria mediante selección adaptativa. Útil para comparar desempeño con igualdad de oportunidades.

```bash
python ipd_simulation_balanced.py
```

### 3. Simulación de Duelo

Dos arquitecturas distintas compiten exclusivamente entre sí. Ideal para estudiar dinámicas de competencia directa.

```bash
python ipd_duel.py
```

## Variables Editables

### En `ipd_simulation.py` (líneas 267-280)

#### 1. Ruta Base de Experimentos

```python
base_folder = "/Users/pauloc/Masters/1 Sem/IA/Proyecto/experiments"
```

Cambia la ruta donde se guardarán los resultados:
```python
base_folder = "./experiments"  # Ruta relativa
```

#### 2. Nombre del Experimento

```python
experiment_path = serializer.create_experiment_folder("ipd_simulation_standard")
```

Personaliza el nombre del experimento:
```python
experiment_path = serializer.create_experiment_folder("mi_experimento_custom")
```

#### 3. Tipos de Jugadores y Arquitecturas

Define la composición inicial de la población. Formato: `(memoria, [capas_ocultas])`

```python
tipos = [
    (2, []),           # Memoria 2, sin capas ocultas (red simple)
    (3, [6]),          # Memoria 3, una capa oculta con 6 neuronas
    (4, [20, 4]),      # Memoria 4, dos capas ocultas con 20 y 4 neuronas
]
```

**Ejemplos de configuración:**

```python
# Configuración simple (solo memoria)
tipos = [
    (2, []),
    (3, []),
    (4, []),
]

# Configuración con arquitecturas profundas
tipos = [
    (2, [8]),
    (3, [16, 8]),
    (4, [32, 16, 8]),
]

# Solo un tipo (para pruebas rápidas)
tipos = [(3, [6])]
```

#### 4. Tamaño de Población Inicial

```python
evolucion.inicializar_poblacion(100, tipos)
```

Cambia el número de jugadores iniciales:
```python
evolucion.inicializar_poblacion(50, tipos)   # Población más pequeña
evolucion.inicializar_poblacion(200, tipos)  # Población más grande
```

#### 5. Número de Generaciones

```python
gens = 200
for gen in range(gens):
    ...
```

Controla cuántas generaciones evolutivas ejecutar:
```python
gens = 50    # Simulación corta (minutos)
gens = 500   # Simulación larga (horas)
```

### Parámetros Implícitos (en el código de clases)

#### Desviación Estándar de Mutación

En `Evolucion.reproducir_poblacion()` (línea 230 en ipd_simulation.py):
```python
def reproducir_poblacion(self, desviacion_std=0.3):
```

Controla la magnitud del ruido gaussiano en la reproducción:
- `0.1` = mutaciones pequeñas (conservador)
- `0.3` = mutaciones moderadas (predeterminado)
- `0.5` = mutaciones grandes (explorador)

Para cambiar globalmente:
```python
evolucion.reproducir_poblacion(desviacion_std=0.2)
```

#### Número de Rondas por Partida

En la clase `Juego.jugar_partido()`:
```python
def jugar_partido(self, jugador1, jugador2, rondas=300):
```

La interacción estándar es de 300 rondas. Para cambiar:
```python
# Necesitarías editar las llamadas internas o pasar como parámetro
# El sistema actual asume 300 rondas fijas
```

#### Top N para Selección

En `Evolucion.seleccion()` (línea 226):
```python
def seleccion(self, puntuaciones, top_n=50):
```

Los mejores 50 jugadores avanzan a la siguiente generación:
```python
def seleccion(self, puntuaciones, top_n=25):  # Selección más restrictiva
def seleccion(self, puntuaciones, top_n=75):  # Selección más laxa
```

## Ejemplo Completo: Personalizar una Simulación

```python
# ipd_simulation.py modificado

base_folder = "./mi_experimento"
serializer = OrganismSerializer(base_folder)
experiment_path = serializer.create_experiment_folder("comparativa_memoria_vs_profundidad")

# Comparar memoria simple vs. profunda
tipos = [
    (2, []),              # Memoria 2 simple
    (2, [16, 8]),         # Memoria 2 profunda
    (4, []),              # Memoria 4 simple
    (4, [32, 16, 8]),     # Memoria 4 profunda
]

evolucion.inicializar_poblacion(200, tipos)  # Población más grande para estudio

gens = 100
for gen in range(gens):
    print(f"Generacion {gen+1}/{gens}...")
    evolucion.ejecutar_generacion(grabador)
    evolucion.guardar_poblacion(serializer, experiment_path)

    # Mostrar progreso intermedio cada 10 generaciones
    if gen % 10 == 0:
        stats = grabador.obtener_estadisticas()
        if gen in stats:
            print(f"  Mejor puntuación: {stats[gen]['maximo']}")
```

## Entender los Resultados

### Estructura de Salida

```
experiments/ipd_simulation_standard_20251202_150000/
├── results/
│   └── resultados.csv           # Datos de todas las generaciones y jugadores
├── organisms/
│   ├── gen0000_player0000.pkl   # Pesos del gen 0, jugador 0
│   ├── gen0000_player0001.pkl
│   └── ...
└── analysis/                     # Carpeta para análisis personalizado
```

### Archivo CSV

`resultados.csv` contiene:
- `generacion`: Número de generación
- `jugador`: ID del jugador
- `tipo_memoria`: Profundidad de memoria (2, 3, 4, etc.)
- `arquitectura`: Configuración de capas ocultas (ej: [6] o [20, 4])
- `puntuacion`: Puntuación total en el torneo

### Interpretación

```python
import pandas as pd

df = pd.read_csv("experiments/.../results/resultados.csv")

# Mejor jugador en la última generación
ultima_gen = df[df['generacion'] == df['generacion'].max()]
mejor = ultima_gen.loc[ultima_gen['puntuacion'].idxmax()]
print(f"Mejor: Memoria {mejor['tipo_memoria']}, Arquitectura {mejor['arquitectura']}")

# Evolución del mejor jugador por generación
mejor_por_gen = df.groupby('generacion')['puntuacion'].max()
print(mejor_por_gen)
```

## Analizar Organismos Evolucionados

Usa el notebook Jupyter `audit_organisms.ipynb` para explorar organismos:

```bash
jupyter notebook audit_organisms.ipynb
```

### Funciones Disponibles

```python
# Inspeccionarel organismo (pesos, sesgos, arquitectura)
inspeccionar_organismo(42)

# Probar decisiones en diferentes situaciones
probar_organismo(42)

# Comparar dos organismos
comparar_organismos(42, 50)

# Simular partida completa entre organismos
jugar_partida(42, 50, rondas=300, mostrar_todo=False)
```

### Reconstruir un Organismo Específico

```python
from organism_serializer import OrganismReconstructior

# Cargar un organismo guardado
org = OrganismReconstructior.reconstruct_from_pickle(
    "experiments/.../organisms/gen0150_player0025.pkl"
)

# Usar en una partida
from ipd_simulation import Juego
juego = Juego()
puntos1, puntos2 = juego.jugar_partido(org, otro_organismo, rondas=300)
```

## Configuración Recomendada para Diferentes Estudios

### Estudio Rápido (5-10 minutos)
```python
gens = 20
evolucion.inicializar_poblacion(50, [(2, []), (3, [4]), (4, [8, 4])])
```

### Estudio Moderado (1-2 horas)
```python
gens = 100
evolucion.inicializar_poblacion(100, [(2, []), (3, [6]), (4, [20, 4])])
```

### Estudio Completo (4-8 horas)
```python
gens = 200
evolucion.inicializar_poblacion(150, [(2, []), (3, [6]), (4, [20, 4])])
```

### Comparativa Arquitecturas (8+ horas)
```python
gens = 300
tipos = [
    (3, []),              # Sin capas ocultas
    (3, [4]),             # Una capa pequeña
    (3, [8]),             # Una capa mediana
    (3, [16]),            # Una capa grande
    (3, [8, 4]),          # Dos capas
    (3, [16, 8, 4]),      # Tres capas
]
evolucion.inicializar_poblacion(200, tipos)
```

## Diferencias Entre Simulaciones

| Característica | Estándar | Balanceada | Duelo |
|---|---|---|---|
| **Competencia** | Todos vs. todos | Todos vs. todos | Población A vs. B |
| **Selección** | Top 50 global | Top proporcional/tipo | Top 25 por población |
| **Riesgo extinción** | Alto | Bajo | Controlado |
| **Uso** | Explorar dinámicas | Comparar tipos equitativamente | Estudiar conflictos directos |

## Notas Importantes

1. **Ruta base**: La ruta `/Users/pauloc/...` está hardcodeada. Actualiza la variable `base_folder` a una ruta local.

2. **Tiempo de ejecución**: Con 200 generaciones y 100 jugadores, la simulación toma ~30-60 minutos dependiendo del hardware.

3. **Reproducibilidad**: Los algoritmos genéticos incluyen aleatoriedad. Usa `np.random.seed()` antes de la simulación si necesitas resultados determinísticos.

4. **Uso de memoria**: Cada organismo guardado ocupa ~1-2 KB. Con 100 jugadores × 200 generaciones = 20,000 organismos ≈ 20-33 MB por experimento.

## Licencia

Este proyecto es parte de una investigación académica de Máster en IA.

## Referencias

- Harrald, B., & Fogel, D. B. (1997). Evolving Strategies for the Prisoner's Dilemma.
- Axelrod, R. (1984). The Evolution of Cooperation.

---

**Última actualización:** Diciembre 2025
