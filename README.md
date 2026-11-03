# Proyecto 1 — Rover autónomo

Implementación reproducible de búsqueda de rutas, metaheurísticas y satisfacción de restricciones (CSP). El proyecto selecciona CSP como representación simbólica. Los ejemplos PDDL y demás documentos del material de curso son referencias y no se distribuyen con este repositorio. No se implementa el apartado opcional de reglas.

## Requisitos

- Python 3.10 o posterior.
- Tkinter 8.6 para la ventana gráfica (incluido en la distribución estándar de Python para Windows).
- No se requieren paquetes externos.

La versión y las dependencias están declaradas en `pyproject.toml` y `requirements.txt`.

## Inicio desde cero en Windows

Desde esta carpeta, en PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest -v test_rover.py
```

El archivo de requisitos no instala paquetes, porque toda la implementación usa la biblioteca estándar. Si se usa otra versión de Python, debe ser 3.10 o posterior.

## Simulador visual

```powershell
python simulador.py
```

La ventana permite fijar semilla, tamaño y cantidad de POI, y escoger BFS, Dijkstra, A* con heurística nula o euclídea, recocido simulado, ascenso de colinas o CSP. **Paso** avanza una expansión, evaluación o decisión; **Reproducir** anima el algoritmo y la ruta. En CSP se muestran asignaciones y podas, y después se ejecuta la misión con la ubicación, batería, memoria y transmisiones visibles. El retardo de animación se puede cambiar en la ventana.

### Qué se ve y qué hace cada modo

El mapa es un grafo: los círculos numerados son ubicaciones y las líneas que los unen son caminos. El color de cada línea y su etiqueta indican el terreno y su costo. **BASE** es el inicio; **COM** es la zona donde se transmiten las muestras; **POI** son objetivos científicos. Las aristas azules resaltan la ruta. Los nodos amarillos son la frontera de búsqueda y los verdes ya fueron expandidos; el punto rojo marca dónde está el rover. Un nodo morado representa un POI ya asignado por el CSP.

BFS, Dijkstra y A* buscan un camino desde la base hasta COM; se anima la frontera de búsqueda y luego el rover recorre el camino hallado. BFS prioriza menos aristas, mientras Dijkstra y A* minimizan el costo del terreno. Ascenso de colinas y recocido simulado comparan órdenes candidatos de visita de los POI; el orden resaltado se recorre como demostración de la optimización, no como una ejecución CSP que valide recursos. CSP, en cambio, construye una misión completa: asigna el orden, decide cuándo comunicar y comprueba batería, memoria, ventanas y plazo. Primero se observan asignaciones, podas y retrocesos; después se ejecutan las acciones de movimiento, recolección, análisis, transmisión y recarga. El panel derecho indica la decisión actual y muestra ubicación, batería, memoria ocupada, POI analizados, muestras transmitidas y tiempo de misión.

## Experimentos reproducibles

```powershell
python rover.py --experimentos-busqueda
python rover.py --experimentos-metaheuristicas
python rover.py --experimentos-csp
python rover.py --demo
```

Las semillas predeterminadas de búsqueda y del CSP general son los enteros 10 a 29; los tamaños son 8, 12 y 16 nodos. Se pueden elegir otros con `--tamanos` y `--semillas`. Los experimentos de búsqueda comparan BFS, Dijkstra, A* con heurística nula y A* con heurística euclídea. Costo y expansiones se calculan una vez por mapa determinista; el tiempo se mide tres veces sobre el mismo mapa.

Hill climbing y recocido simulado se ejecutan con 200 evaluaciones por corrida, ocho corridas por semilla y los mismos mapas y semillas derivadas. Se comparan con vecino más cercano. El CSP compara asignación cronológica y selección MRV/LCV, ambas con forward checking. Sus dominios finitos incluyen la visita, la decisión de cerrar el lote y la ventana de comunicación. El orden de POI puede pasarse al CSP como preferencia, nunca como restricción: el solver conserva la posibilidad de cambiarlo para satisfacer recursos y plazos. Las tablas reportan media y desviación estándar, no la mejor corrida.

El comando `--experimentos-csp` ejecuta además un conjunto de estrés fijo sobre 20 semillas (0 a 19): mapas de 8 nodos y 5 POI, batería 9.0, memoria para dos muestras, plazo 25 y ventanas de comunicación de seis unidades separadas por nueve unidades. Reporta la cantidad de misiones factibles y, entre ellas, asignaciones probadas, retrocesos, costo, duración de misión y tiempo de cómputo. Las instancias no factibles también se cuentan en la tasa de factibilidad; con la configuración reportada, ambas estrategias encuentran plan en 17 de las 20. El solver admite como máximo 12 POI; el conjunto de estrés no supera ese límite.

Ejemplo de parámetros alternativos:

```powershell
python rover.py --experimentos-busqueda --tamanos 8 12 --semillas 7 13 21
python rover.py --experimentos-metaheuristicas --presupuesto 300 --repeticiones-meta 10
```

## Modelo y archivos

- `rover.py`: generador por semilla; búsqueda ponderada; búsqueda metaheurística del orden de POI; CSP, restricciones y protocolo experimental.
- `simulador.py`: interfaz gráfica Tkinter y animación de los motores de `rover.py`.
- `test_rover.py`: pruebas de reproducibilidad, optimalidad de rutas, presupuesto metaheurístico, CSP y ejecución de recursos.
- `Informe rover.tex`: informe LaTeX con protocolo, resultados, análisis y límites.
- `Informe rover.pdf`: informe compilado listo para consulta.
- Los ejemplos PDDL del material de curso no se ejecutan ni se incluyen; el módulo simbólico elegido y reportado en este proyecto es CSP.

El entorno se genera como grafo no dirigido y conexo, con terrenos de costo 1.0, 1.7, 2.4 y 3.1. El rover dispone de batería 15.0, memoria para dos muestras y ventanas de comunicación repetidas hasta el límite de misión de 200 unidades. El CSP admite hasta 12 POI para contener el crecimiento combinatorio.
