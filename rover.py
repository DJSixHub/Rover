#!/usr/bin/env python3
"""Modelos reproducibles de búsqueda, metaheurísticas y CSP para el rover."""

from __future__ import annotations

#####SECCION 1: IMPORTACIONES, CONSTANTES Y ESTRUCTURAS
import argparse
import heapq
import math
import random
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Sequence, Tuple

TERRAIN_COSTS = {
    "llano": 1.0,
    "roca": 1.7,
    "arena": 2.4,
    "duna": 3.1,
}
SEARCH_ALGORITHMS = ("bfs", "dijkstra", "astar_zero", "astar_euclidean")
DEFAULT_SEEDS = tuple(range(10, 30))
ENERGY_SCALE = 10
BATTERY_CAPACITY = 150
MEMORY_CAPACITY = 2
MISSION_TIME_LIMIT = 200.0
COMMUNICATION_WINDOWS = tuple(
    (8.0 + 20.0 * index, 20.0 + 20.0 * index) for index in range(10)
)
TAKE_DURATION = 0.2
ANALYZE_DURATION = 0.2
TRANSMIT_DURATION = 0.4
RECHARGE_DURATION = 0.8
CSP_MAX_NODES = 100_000
CSP_MAX_POIS = 12
CSP_STRESS_SEEDS = tuple(range(20))
CSP_STRESS_WINDOWS = tuple(
    (5.0 + 15.0 * index, 11.0 + 15.0 * index) for index in range(10)
)
CSP_STRESS_BATTERY = 90
CSP_STRESS_DEADLINE = 25.0


@dataclass
class RoverInstance:
    """Grafo ponderado del entorno y los elementos de la misión."""

    nombre: str
    n: int
    base: int
    zona_comunicacion: int
    puntos_interes: List[int]
    posiciones: Dict[int, Tuple[float, float]]
    grafo: Dict[int, List[Tuple[int, float, str]]] = field(default_factory=dict)
    seed: int = 0
    ventanas_comunicacion: Tuple[Tuple[float, float], ...] = COMMUNICATION_WINDOWS

    def __post_init__(self) -> None:
        for nodo in range(self.n):
            self.grafo.setdefault(nodo, [])

    def vecinos(self, nodo: int) -> Sequence[Tuple[int, float, str]]:
        return self.grafo[nodo]


@dataclass
class SearchResult:
    """Métricas y ruta devueltas por un algoritmo de búsqueda."""

    algoritmo: str
    camino: List[int]
    costo: float
    nodos_expandidos: int
    nodos_generados: int
    tiempo_ms: float


@dataclass(frozen=True)
class SearchEvent:
    """Un paso de expansión para la animación y el seguimiento experimental."""

    nodo: int
    expandidos: Tuple[int, ...]
    frontera: Tuple[int, ...]
    cantidad: int


@dataclass
class MetaheuristicResult:
    """Mejor orden de visita encontrado por una metaheurística."""

    algoritmo: str
    orden: List[int]
    costo: float
    evaluaciones: int
    tiempo_ms: float


@dataclass(frozen=True)
class MetaheuristicEvent:
    """Estado de una iteración de búsqueda local."""

    orden_actual: Tuple[int, ...]
    mejor_orden: Tuple[int, ...]
    costo_actual: float
    mejor_costo: float
    evaluacion: int


@dataclass
class CSPResult:
    """Asignación CSP factible y métricas de búsqueda."""

    orden: List[int]
    comunicar_despues_de: List[int]
    ventana_por_lote: List[int]
    acciones: List[str]
    costo_ruta: float
    tiempo_mision: float
    nodos_explorados: int
    retrocesos: int


@dataclass(frozen=True)
class CSPExperimentStats:
    """Resumen de ejecuciones CSP sobre una configuración experimental."""

    estrategia: str
    instancias: int
    factibles: int
    nodos_media: float
    nodos_desv: float
    retrocesos_media: float
    retrocesos_desv: float
    costo_media: float
    costo_desv: float
    mision_media: float
    mision_desv: float
    computo_media: float
    computo_desv: float


@dataclass(frozen=True)
class CSPEvent:
    """Evento de selección, propagación o retroceso de la búsqueda CSP."""

    tipo: str
    detalle: str
    asignados: Tuple[Tuple[int, int, bool, int], ...]
    dominios_restantes: int


@dataclass
class _ScheduleTrace:
    acciones: List[str]
    costo_ruta: float
    tiempo_mision: float


#####SECCION 2: GENERACION PARAMETRIZABLE DEL ENTORNO
def euclidea(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def generar_instancia(seed: int, tamano: int = 10, num_pois: int = 3) -> RoverInstance:
    """Genera un mapa conexo con terrenos y POI a partir de una semilla."""
    if tamano < 3:
        raise ValueError("El mapa necesita al menos tres nodos.")
    if num_pois < 0:
        raise ValueError("num_pois no puede ser negativo.")

    rnd = random.Random(seed)
    posiciones = {
        nodo: (rnd.uniform(0.0, 1.0), rnd.uniform(0.0, 1.0))
        for nodo in range(tamano)
    }
    base = 0
    zona = max(
        range(1, tamano),
        key=lambda nodo: (euclidea(posiciones[nodo], posiciones[base]), nodo),
    )
    disponibles = [nodo for nodo in range(tamano) if nodo not in {base, zona}]
    pois = sorted(rnd.sample(disponibles, k=min(num_pois, len(disponibles))))
    grafo: Dict[int, List[Tuple[int, float, str]]] = defaultdict(list)

    def conectar(origen: int, destino: int, terreno: str) -> None:
        costo = TERRAIN_COSTS[terreno]
        grafo[origen].append((destino, costo, terreno))
        grafo[destino].append((origen, costo, terreno))

    for origen in range(tamano):
        for destino in range(origen + 1, tamano):
            if rnd.random() < 0.32:
                terreno = rnd.choices(
                    population=list(TERRAIN_COSTS),
                    weights=[0.45, 0.25, 0.2, 0.1],
                    k=1,
                )[0]
                conectar(origen, destino, terreno)

    for nodo in range(tamano - 1):
        if not any(vecino == nodo + 1 for vecino, _, _ in grafo[nodo]):
            conectar(nodo, nodo + 1, "llano")

    return RoverInstance(
        nombre=f"instancia-seed-{seed}-n{tamano}",
        n=tamano,
        base=base,
        zona_comunicacion=zona,
        puntos_interes=pois,
        posiciones=posiciones,
        grafo=grafo,
        seed=seed,
    )


#####SECCION 3: BUSQUEDA DE RUTAS Y HEURISTICAS
def _validar_extremos(instance: RoverInstance, inicio: int, meta: int) -> None:
    if inicio not in instance.grafo or meta not in instance.grafo:
        raise ValueError(f"Nodo de inicio o meta inexistente: {inicio} -> {meta}.")


def _factor_heuristica(instance: RoverInstance) -> float:
    razones = []
    for origen, vecinos in instance.grafo.items():
        for destino, costo, _ in vecinos:
            distancia = euclidea(instance.posiciones[origen], instance.posiciones[destino])
            if distancia > 0.0:
                razones.append(costo / distancia)
    return min(razones) if razones else 0.0


def _reconstruir_camino(
    padres: Dict[int, Optional[int]], inicio: int, meta: int
) -> List[int]:
    camino = [meta]
    while camino[-1] != inicio:
        padre = padres.get(camino[-1])
        if padre is None:
            raise RuntimeError("La cadena de predecesores está incompleta.")
        camino.append(padre)
    camino.reverse()
    return camino


def trazar_busqueda(
    instance: RoverInstance, inicio: int, meta: int, algoritmo: str
) -> Generator[SearchEvent, None, SearchResult]:
    """Expone una expansión por paso y devuelve la búsqueda completa al terminar."""
    algoritmo = algoritmo.lower()
    if algoritmo == "astar":
        algoritmo = "astar_euclidean"
    if algoritmo not in SEARCH_ALGORITHMS:
        raise ValueError(f"Algoritmo no soportado: {algoritmo}.")
    _validar_extremos(instance, inicio, meta)
    t0 = time.perf_counter()
    padres: Dict[int, Optional[int]] = {inicio: None}
    expandidos: List[int] = []
    visitados: set[int] = {inicio}
    cantidad_generados = 1

    if algoritmo == "bfs":
        frontera_bfs = deque([inicio])
        while frontera_bfs:
            actual = frontera_bfs.popleft()
            expandidos.append(actual)
            if actual == meta:
                yield SearchEvent(actual, tuple(expandidos), tuple(frontera_bfs), len(expandidos))
                break
            for vecino, _, _ in instance.vecinos(actual):
                if vecino not in visitados:
                    visitados.add(vecino)
                    padres[vecino] = actual
                    frontera_bfs.append(vecino)
                    cantidad_generados += 1
            yield SearchEvent(actual, tuple(expandidos), tuple(frontera_bfs), len(expandidos))
        else:
            raise ValueError(f"No existe ruta entre los nodos {inicio} y {meta}.")
    else:
        escala = (
            _factor_heuristica(instance)
            if algoritmo == "astar_euclidean"
            else 0.0
        )

        def estimar(nodo: int) -> float:
            return escala * euclidea(instance.posiciones[nodo], instance.posiciones[meta])

        heap: List[Tuple[float, float, int]] = [(estimar(inicio), 0.0, inicio)]
        mejores = {inicio: 0.0}
        while heap:
            _, costo_actual, actual = heapq.heappop(heap)
            if costo_actual > mejores.get(actual, math.inf):
                continue
            expandidos.append(actual)
            if actual == meta:
                yield SearchEvent(
                    actual,
                    tuple(expandidos),
                    tuple(item[2] for item in heap),
                    len(expandidos),
                )
                break
            for vecino, costo, _ in instance.vecinos(actual):
                nuevo_costo = costo_actual + costo
                if nuevo_costo < mejores.get(vecino, math.inf):
                    mejores[vecino] = nuevo_costo
                    padres[vecino] = actual
                    visitados.add(vecino)
                    heapq.heappush(
                        heap,
                        (nuevo_costo + estimar(vecino), nuevo_costo, vecino),
                    )
                    cantidad_generados += 1
            yield SearchEvent(
                actual,
                tuple(expandidos),
                tuple(item[2] for item in heap),
                len(expandidos),
            )
        else:
            raise ValueError(f"No existe ruta entre los nodos {inicio} y {meta}.")

    camino = _reconstruir_camino(padres, inicio, meta)
    costo_total = 0.0
    for origen, destino in zip(camino, camino[1:]):
        arista = next(
            (costo for vecino, costo, _ in instance.vecinos(origen) if vecino == destino),
            None,
        )
        if arista is None:
            raise RuntimeError(f"Falta la arista {origen} -> {destino}.")
        costo_total += arista
    return SearchResult(
        algoritmo=algoritmo,
        camino=camino,
        costo=costo_total,
        nodos_expandidos=len(expandidos),
        nodos_generados=cantidad_generados,
        tiempo_ms=(time.perf_counter() - t0) * 1000.0,
    )


def buscar_ruta(
    instance: RoverInstance, inicio: int, meta: int, algoritmo: str
) -> SearchResult:
    """Resuelve completamente una búsqueda; comparte el motor usado por la GUI."""
    pasos = trazar_busqueda(instance, inicio, meta, algoritmo)
    while True:
        try:
            next(pasos)
        except StopIteration as resultado:
            return resultado.value


#####SECCION 4: METAHEURISTICAS PARA EL ORDEN DE VISITA
def _matriz_distancias(instance: RoverInstance) -> Dict[Tuple[int, int], float]:
    distancias = {}
    for origen in range(instance.n):
        for destino in range(instance.n):
            distancias[origen, destino] = buscar_ruta(
                instance, origen, destino, "dijkstra"
            ).costo
    return distancias


def costo_orden_visita(
    instance: RoverInstance,
    orden: Sequence[int],
    distancias: Optional[Dict[Tuple[int, int], float]] = None,
    memoria: int = MEMORY_CAPACITY,
) -> float:
    """Costo del tour con transmisión al alcanzar el límite de memoria."""
    if len(orden) != len(instance.puntos_interes) or set(orden) != set(
        instance.puntos_interes
    ):
        raise ValueError("El orden debe ser una permutación exacta de los POI.")
    if memoria < 1:
        raise ValueError("La memoria debe ser al menos una muestra.")
    d = distancias or _matriz_distancias(instance)
    costo_total = 0.0
    actual = instance.base
    for indice, poi in enumerate(orden):
        costo_total += d[actual, poi]
        actual = poi
        if (indice + 1) % memoria == 0 or indice == len(orden) - 1:
            costo_total += d[actual, instance.zona_comunicacion]
            actual = instance.zona_comunicacion
    return costo_total


def orden_visita_vecino_mas_cercano(
    instance: RoverInstance,
    distancias: Optional[Dict[Tuple[int, int], float]] = None,
    memoria: int = MEMORY_CAPACITY,
) -> List[int]:
    """Línea base greedy, retornando a comunicar cuando se llena la memoria."""
    d = distancias or _matriz_distancias(instance)
    restantes = set(instance.puntos_interes)
    orden: List[int] = []
    actual = instance.base
    while restantes:
        poi = min(restantes, key=lambda candidato: (d[actual, candidato], candidato))
        orden.append(poi)
        restantes.remove(poi)
        actual = poi
        if len(orden) % memoria == 0 or not restantes:
            actual = instance.zona_comunicacion
    return orden


def trazar_metaheuristica(
    instance: RoverInstance,
    algoritmo: str,
    seed: int,
    presupuesto: int = 200,
    memoria: int = MEMORY_CAPACITY,
) -> Generator[MetaheuristicEvent, None, MetaheuristicResult]:
    """Genera pasos visualizables de recocido simulado o ascenso aleatorio."""
    algoritmo = algoritmo.lower()
    if algoritmo not in {"simulated_annealing", "hill_climbing"}:
        raise ValueError(f"Metaheurística no soportada: {algoritmo}.")
    if presupuesto < 1:
        raise ValueError("El presupuesto debe ser al menos una evaluación.")
    t0 = time.perf_counter()
    rnd = random.Random(seed)
    d = _matriz_distancias(instance)
    actual = list(instance.puntos_interes)
    rnd.shuffle(actual)
    costo_actual = costo_orden_visita(instance, actual, d, memoria)
    mejor = actual[:]
    costo_mejor = costo_actual
    yield MetaheuristicEvent(
        tuple(actual), tuple(mejor), costo_actual, costo_mejor, 1
    )

    evaluacion = 1
    while evaluacion < presupuesto and len(actual) >= 2:
        i, j = sorted(rnd.sample(range(len(actual)), 2))
        candidato = actual[:]
        if algoritmo == "simulated_annealing":
            candidato[i : j + 1] = reversed(candidato[i : j + 1])
        else:
            candidato[i], candidato[j] = candidato[j], candidato[i]
        costo_candidato = costo_orden_visita(instance, candidato, d, memoria)
        evaluacion += 1

        if algoritmo == "simulated_annealing":
            fraccion = evaluacion / presupuesto
            temperatura = max(0.01, 2.0 * (1.0 - fraccion))
            delta = costo_candidato - costo_actual
            acepta = delta < 0.0 or rnd.random() < math.exp(-delta / temperatura)
        else:
            acepta = costo_candidato < costo_actual

        if acepta:
            actual = candidato
            costo_actual = costo_candidato
        if costo_candidato < costo_mejor:
            mejor = candidato[:]
            costo_mejor = costo_candidato

        yield MetaheuristicEvent(
            tuple(actual),
            tuple(mejor),
            costo_actual,
            costo_mejor,
            evaluacion,
        )

    return MetaheuristicResult(
        algoritmo=algoritmo,
        orden=mejor,
        costo=costo_mejor,
        evaluaciones=evaluacion,
        tiempo_ms=(time.perf_counter() - t0) * 1000.0,
    )


def optimizar_orden(
    instance: RoverInstance,
    algoritmo: str,
    seed: int,
    presupuesto: int = 200,
    memoria: int = MEMORY_CAPACITY,
) -> MetaheuristicResult:
    """Ejecuta y devuelve la mejor solución de una metaheurística."""
    pasos = trazar_metaheuristica(instance, algoritmo, seed, presupuesto, memoria)
    while True:
        try:
            next(pasos)
        except StopIteration as resultado:
            return resultado.value


#####SECCION 5: CSP PARA ORDEN, ACCIONES, MEMORIA, BATERIA Y TIEMPO
def _energia_arista(costo: float) -> int:
    return math.ceil(costo * ENERGY_SCALE - 1e-9)


def _ruta_y_consumo(
    instance: RoverInstance,
    origen: int,
    destino: int,
    rutas: Optional[Dict[Tuple[int, int], Tuple[List[int], float]]] = None,
) -> Tuple[List[int], float]:
    if rutas is not None and (origen, destino) in rutas:
        return rutas[origen, destino]
    ruta = buscar_ruta(instance, origen, destino, "dijkstra")
    return ruta.camino, ruta.costo


def _ventana_asignada(
    ventanas: Sequence[Tuple[float, float]],
    indice: int,
    llegada: float,
    servicio: float,
) -> Optional[Tuple[float, float]]:
    if indice < 0 or indice >= len(ventanas):
        return None
    inicio, fin = ventanas[indice]
    comienzo = max(llegada, inicio)
    if comienzo + servicio > fin:
        return None
    return comienzo, fin


def _simular_prefijo_csp(
    instance: RoverInstance,
    decisiones: Sequence[Tuple[int, bool, int]],
    capacidad_bateria: int,
    capacidad_memoria: int,
    limite_tiempo: float,
    rutas: Optional[Dict[Tuple[int, int], Tuple[List[int], float]]] = None,
) -> Optional[_ScheduleTrace]:
    """Evalúa restricciones de recursos y construye acciones para un prefijo."""
    acciones: List[str] = []
    origen = instance.base
    bateria = capacidad_bateria
    tiempo = 0.0
    costo_ruta = 0.0
    lote: List[int] = []

    def desplazarse(destino: int) -> bool:
        nonlocal origen, bateria, tiempo, costo_ruta
        camino, costo = _ruta_y_consumo(instance, origen, destino, rutas)
        estaciones = {instance.base, instance.zona_comunicacion}
        if origen in estaciones and bateria < capacidad_bateria:
            acciones.append("recharge")
            tiempo += RECHARGE_DURATION
            bateria = capacidad_bateria
        for indice, (u, v) in enumerate(zip(camino, camino[1:])):
            energia = _energia_arista(
                next(c for vecino, c, _ in instance.vecinos(u) if vecino == v)
            )
            if energia > bateria and u in estaciones:
                acciones.append("recharge")
                tiempo += RECHARGE_DURATION
                bateria = capacidad_bateria
            if energia > bateria:
                return False
            acciones.append(f"move {u} {v}")
            bateria -= energia
            if v in estaciones and v != destino and indice < len(camino) - 2:
                acciones.append("recharge")
                tiempo += RECHARGE_DURATION
                bateria = capacidad_bateria
        origen = destino
        tiempo += costo
        costo_ruta += costo
        return True

    for poi, cerrar_lote, indice_ventana in decisiones:
        if poi in lote or poi not in instance.puntos_interes:
            return None
        if len(lote) >= capacidad_memoria:
            return None
        if not desplazarse(poi):
            return None
        acciones.append(f"take {poi}")
        tiempo += TAKE_DURATION
        acciones.append(f"analyze {poi}")
        tiempo += ANALYZE_DURATION
        lote.append(poi)
        if cerrar_lote and lote:
            if not desplazarse(instance.zona_comunicacion):
                return None
            servicio = len(lote) * TRANSMIT_DURATION
            ventana = _ventana_asignada(
                instance.ventanas_comunicacion, indice_ventana, tiempo, servicio
            )
            if ventana is None:
                return None
            inicio_servicio, fin_ventana = ventana
            if inicio_servicio > tiempo:
                acciones.append(f"wait {inicio_servicio - tiempo:.2f}")
                tiempo = inicio_servicio
            for muestra in lote:
                acciones.append(f"transmit {muestra} {instance.zona_comunicacion}")
                tiempo += TRANSMIT_DURATION
            if tiempo > fin_ventana + 1e-9:
                return None
            lote.clear()
            if bateria < capacidad_bateria:
                acciones.append("recharge")
                tiempo += RECHARGE_DURATION
                bateria = capacidad_bateria
        elif indice_ventana != -1:
            return None
        if tiempo > limite_tiempo:
            return None

    return _ScheduleTrace(acciones, costo_ruta, tiempo)


def trazar_resolucion_csp(
    instance: RoverInstance,
    capacidad_bateria: int = BATTERY_CAPACITY,
    capacidad_memoria: int = MEMORY_CAPACITY,
    limite_tiempo: float = MISSION_TIME_LIMIT,
    max_nodos: int = CSP_MAX_NODES,
    usar_heuristicas: bool = True,
    orden_preferido: Optional[Sequence[int]] = None,
) -> Generator[CSPEvent, None, CSPResult]:
    """Resuelve el CSP con forward checking, MRV y LCV.

    X_i = (POI, cierre del lote, índice de ventana de comunicación).
    El índice de ventana es una variable temporal discreta; la hora exacta de
    servicio se obtiene al propagar los tiempos de viaje y espera.
    """
    if capacidad_bateria <= 0 or capacidad_memoria <= 0:
        raise ValueError("Batería y memoria deben ser positivas.")
    if limite_tiempo <= 0.0 or max_nodos <= 0:
        raise ValueError("El plazo y el máximo de nodos deben ser positivos.")
    cantidad = len(instance.puntos_interes)
    if cantidad == 0:
        return CSPResult([], [], [], [], 0.0, 0.0, 0, 0)
    if cantidad > CSP_MAX_POIS:
        raise ValueError(
            f"El CSP admite como máximo {CSP_MAX_POIS} puntos de interés."
        )
    if orden_preferido is not None and (
        len(orden_preferido) != cantidad
        or set(orden_preferido) != set(instance.puntos_interes)
    ):
        raise ValueError("orden_preferido debe ser una permutación exacta de los POI.")
    rango_preferido = (
        {poi: posicion for posicion, poi in enumerate(orden_preferido)}
        if orden_preferido is not None
        else {}
    )
    if capacidad_memoria < 1:
        raise ValueError("La memoria debe admitir al menos una muestra.")

    asignacion: Dict[int, Tuple[int, bool, int]] = {}
    explorados = 0
    retrocesos = 0
    rutas = {
        (origen, destino): (
            resultado.camino,
            resultado.costo,
        )
        for origen in range(instance.n)
        for destino in range(instance.n)
        for resultado in [
            buscar_ruta(instance, origen, destino, "dijkstra")
        ]
    }
    yield CSPEvent(
        "inicio",
        f"{cantidad} variables X_i=(POI, cierre, ventana temporal); "
        "forward checking inicializado",
        (),
        cantidad * len(instance.puntos_interes) * (1 + len(instance.ventanas_comunicacion)),
    )

    def prefijo_contiguo() -> List[Tuple[int, bool, int]]:
        resultado = []
        for posicion in range(cantidad):
            if posicion not in asignacion:
                break
            resultado.append(asignacion[posicion])
        return resultado

    def valores_dominio(
        posicion: int, usados: set[int]
    ) -> List[Tuple[int, bool, int]]:
        valores: List[Tuple[int, bool, int]] = []
        prefijo = prefijo_contiguo()
        es_siguiente = posicion == len(prefijo)
        lote_actual = 0
        if es_siguiente:
            for _, cierre, _ in reversed(prefijo):
                if cierre:
                    break
                lote_actual += 1

        for poi in instance.puntos_interes:
            if poi in usados:
                continue
            debe_cerrar = (
                es_siguiente and lote_actual + 1 >= capacidad_memoria
            ) or posicion == cantidad - 1
            cierres = (True,) if debe_cerrar else (False, True)
            for cerrar in cierres:
                indices_ventana = (
                    range(len(instance.ventanas_comunicacion))
                    if cerrar
                    else (-1,)
                )
                for indice_ventana in indices_ventana:
                    valor = (poi, cerrar, indice_ventana)
                    if es_siguiente:
                        asignacion[posicion] = valor
                        factible = (
                            _simular_prefijo_csp(
                                instance,
                                prefijo_contiguo(),
                                capacidad_bateria,
                                capacidad_memoria,
                                limite_tiempo,
                                rutas,
                            )
                            is not None
                        )
                        del asignacion[posicion]
                        if not factible:
                            continue
                    valores.append(valor)
                    if cerrar and es_siguiente:
                        # Una ventana posterior solo añade espera y no puede
                        # mejorar batería, memoria ni plazo.
                        break
        return valores

    def dominios_actuales() -> Dict[int, List[Tuple[int, bool, int]]]:
        usados = {poi for poi, _, _ in asignacion.values()}
        dominios = {}
        for posicion in range(cantidad):
            if posicion not in asignacion:
                dominios[posicion] = valores_dominio(posicion, usados)
        return dominios

    def propagar_y_validar() -> Tuple[bool, int]:
        prefijo = prefijo_contiguo()
        traza = _simular_prefijo_csp(
            instance,
            prefijo,
            capacidad_bateria,
            capacidad_memoria,
            limite_tiempo,
            rutas,
        )
        if traza is None:
            return False, 0
        dominios = dominios_actuales()
        if any(not dominio for dominio in dominios.values()):
            return False, sum(map(len, dominios.values()))
        return True, sum(map(len, dominios.values()))

    def resolver() -> Generator[CSPEvent, None, Optional[_ScheduleTrace]]:
        nonlocal explorados, retrocesos
        if len(asignacion) == cantidad:
            decisiones = [asignacion[i] for i in range(cantidad)]
            traza = _simular_prefijo_csp(
                instance,
                decisiones,
                capacidad_bateria,
                capacidad_memoria,
                limite_tiempo,
                rutas,
            )
            return traza
        if explorados >= max_nodos:
            raise RuntimeError(f"El CSP alcanzó el límite de {max_nodos} nodos.")

        dominios = dominios_actuales()
        if not dominios:
            return None
        if usar_heuristicas:
            posicion = min(
                dominios,
                key=lambda i: (
                    len({poi for poi, _, _ in dominios[i]}),
                    i,
                ),
            )
        else:
            posicion = min(dominios)
        dominio = dominios[posicion]
        if not dominio:
            retrocesos += 1
            return None

        prefijo = prefijo_contiguo()
        origen = (
            instance.base
            if not prefijo
            else (
                instance.zona_comunicacion
                if prefijo[-1][1]
                else prefijo[-1][0]
            )
        )

        def puntuacion_lcv(
            valor: Tuple[int, bool, int],
        ) -> Tuple[int, int, float, int, bool, int]:
            poi, cerrar, indice_ventana = valor
            asignacion[posicion] = valor
            prefijo_prueba = prefijo_contiguo()
            siguiente = len(prefijo_prueba)
            opciones_siguientes = 0
            if siguiente < cantidad and siguiente not in asignacion:
                usados = {p for p, _, _ in asignacion.values()}
                opciones_siguientes = len(valores_dominio(siguiente, usados))
            del asignacion[posicion]
            distancia = rutas[origen, poi][1]
            return (
                -opciones_siguientes,
                rango_preferido.get(poi, 0),
                distancia,
                poi,
                cerrar,
                indice_ventana,
            )

        candidatos = sorted(dominio, key=puntuacion_lcv) if usar_heuristicas else sorted(dominio)
        yield CSPEvent(
            "seleccion",
            f"X_{posicion}: MRV selecciona {len(dominio)} valores candidatos"
            if usar_heuristicas
            else f"X_{posicion}: selección cronológica",
            tuple(
                (i, poi, cierre, ventana)
                for i, (poi, cierre, ventana) in sorted(asignacion.items())
            ),
            sum(map(len, dominios.values())),
        )

        for valor in candidatos:
            explorados += 1
            asignacion[posicion] = valor
            consistente, cantidad_dominios = propagar_y_validar()
            estado = tuple(
                (i, poi, cierre, ventana)
                for i, (poi, cierre, ventana) in sorted(asignacion.items())
            )
            if consistente:
                yield CSPEvent(
                    "asignacion",
                    f"POI {valor[0]} en posición {posicion}; "
                    f"{'comunicar en ventana ' + str(valor[2]) if valor[1] else 'continuar lote'}",
                    estado,
                    cantidad_dominios,
                )
                solucion = yield from resolver()
                if solucion is not None:
                    return solucion
            else:
                yield CSPEvent(
                    "poda",
                    f"Se descarta X_{posicion}={valor}: recurso, ventana, plazo "
                    "o dominio futuro inconsistente",
                    estado,
                    cantidad_dominios,
                )
            del asignacion[posicion]
            retrocesos += 1
            yield CSPEvent(
                "retroceso",
                f"Retroceso en X_{posicion}; se restaura el dominio",
                tuple(
                    (i, p, c, ventana)
                    for i, (p, c, ventana) in sorted(asignacion.items())
                ),
                cantidad_dominios,
            )
        return None

    traza = yield from resolver()
    if traza is None:
        raise RuntimeError("El CSP no encontró una misión factible.")
    decisiones = [asignacion[i] for i in range(cantidad)]
    return CSPResult(
        orden=[poi for poi, _, _ in decisiones],
        comunicar_despues_de=[
            posicion
            for posicion, (_, cerrar, _) in enumerate(decisiones)
            if cerrar
        ],
        ventana_por_lote=[
            ventana
            for _, cerrar, ventana in decisiones
            if cerrar
        ],
        acciones=traza.acciones,
        costo_ruta=traza.costo_ruta,
        tiempo_mision=traza.tiempo_mision,
        nodos_explorados=explorados,
        retrocesos=retrocesos,
    )


def resolver_csp(
    instance: RoverInstance,
    capacidad_bateria: int = BATTERY_CAPACITY,
    capacidad_memoria: int = MEMORY_CAPACITY,
    limite_tiempo: float = MISSION_TIME_LIMIT,
    usar_heuristicas: bool = True,
    orden_preferido: Optional[Sequence[int]] = None,
) -> CSPResult:
    """Resuelve el CSP sin exponer los eventos de la interfaz gráfica."""
    pasos = trazar_resolucion_csp(
        instance,
        capacidad_bateria,
        capacidad_memoria,
        limite_tiempo,
        usar_heuristicas=usar_heuristicas,
        orden_preferido=orden_preferido,
    )
    while True:
        try:
            next(pasos)
        except StopIteration as resultado:
            return resultado.value


#####SECCION 6: EXPERIMENTOS REPRODUCIBLES Y ESTADISTICA
def _resumen(valores: Sequence[float]) -> Tuple[float, float]:
    if not valores:
        raise ValueError("No se puede resumir una colección vacía.")
    return statistics.mean(valores), statistics.pstdev(valores)


def ejecutar_experimentos_busqueda(
    tamanos: Sequence[int] = (8, 12, 16),
    semillas: Sequence[int] = DEFAULT_SEEDS,
    repeticiones_tiempo: int = 3,
) -> List[Tuple[str, str, float, float, float, float, float, float]]:
    """Compara algoritmos y heurísticas sobre instancias compartidas."""
    if not tamanos or not semillas or repeticiones_tiempo < 1:
        raise ValueError("Se requieren tamaños, semillas y al menos una medición.")
    resultados = []
    for tamano in tamanos:
        metricas = {alg: [[], [], []] for alg in SEARCH_ALGORITHMS}
        for semilla in semillas:
            instancia = generar_instancia(semilla, tamano, num_pois=min(5, tamano - 2))
            for algoritmo in SEARCH_ALGORITHMS:
                mediciones = [
                    buscar_ruta(
                        instancia,
                        instancia.base,
                        instancia.zona_comunicacion,
                        algoritmo,
                    )
                    for _ in range(repeticiones_tiempo)
                ]
                primera = mediciones[0]
                metricas[algoritmo][0].append(
                    statistics.mean(r.tiempo_ms for r in mediciones)
                )
                metricas[algoritmo][1].append(float(primera.nodos_expandidos))
                metricas[algoritmo][2].append(primera.costo)
        for algoritmo, (tiempos, nodos, costos) in metricas.items():
            tm, td = _resumen(tiempos)
            nm, nd = _resumen(nodos)
            cm, cd = _resumen(costos)
            resultados.append((str(tamano), algoritmo, tm, td, nm, nd, cm, cd))
    return resultados


def ejecutar_experimentos_metaheuristicas(
    tamanos: Sequence[int] = (8, 12, 16),
    semillas: Sequence[int] = DEFAULT_SEEDS,
    presupuesto: int = 200,
    repeticiones: int = 8,
) -> List[Tuple[str, float, float, float, float, float, float, float, float]]:
    """Compara dos metaheurísticas con igual presupuesto de evaluaciones."""
    if not tamanos or not semillas or presupuesto < 1 or repeticiones < 1:
        raise ValueError("Parámetros experimentales inválidos.")
    resultados = []
    for tamano in tamanos:
        por_algoritmo: Dict[str, List[float]] = {
            "hill_climbing": [],
            "simulated_annealing": [],
        }
        mejoras: Dict[str, List[float]] = {
            "hill_climbing": [],
            "simulated_annealing": [],
        }
        costos_base = []
        tiempos = {"hill_climbing": [], "simulated_annealing": []}
        for semilla in semillas:
            cantidad_pois = min(tamano - 2, max(6, tamano // 2))
            instancia = generar_instancia(semilla, tamano, num_pois=cantidad_pois)
            distancias = _matriz_distancias(instancia)
            baseline = orden_visita_vecino_mas_cercano(instancia, distancias)
            costos_base.append(costo_orden_visita(instancia, baseline, distancias))
            for repeticion in range(repeticiones):
                for algoritmo in ("hill_climbing", "simulated_annealing"):
                    resultado = optimizar_orden(
                        instancia,
                        algoritmo,
                        seed=semilla * 10_007 + repeticion,
                        presupuesto=presupuesto,
                    )
                    por_algoritmo[algoritmo].append(resultado.costo)
                    mejoras[algoritmo].append(
                        100.0
                        * (costos_base[-1] - resultado.costo)
                        / costos_base[-1]
                    )
                    tiempos[algoritmo].append(resultado.tiempo_ms)
        base_mean, base_std = _resumen(costos_base)
        for algoritmo, costos in por_algoritmo.items():
            cm, cd = _resumen(costos)
            im, idesv = _resumen(mejoras[algoritmo])
            tm, td = _resumen(tiempos[algoritmo])
            resultados.append(
                (
                    f"{tamano}/{algoritmo}",
                    base_mean,
                    base_std,
                    cm,
                    cd,
                    im,
                    idesv,
                    tm,
                    td,
                )
            )
    return resultados


def ejecutar_experimentos_csp(
    tamanos: Sequence[int] = (8, 12, 16),
    semillas: Sequence[int] = DEFAULT_SEEDS,
) -> List[Tuple[str, float, float, float, float, float, float, float, float, float, float]]:
    """Mide tiempo y trabajo del CSP con heurísticas frente a orden fijo."""
    if not tamanos or not semillas:
        raise ValueError("Se requieren tamaños y semillas.")
    resultados = []
    for tamano in tamanos:
        for heuristicas in (False, True):
            nodos, retrocesos, tiempos, costos, computo = [], [], [], [], []
            for semilla in semillas:
                cantidad_pois = min(tamano - 2, max(4, tamano // 2))
                instancia = generar_instancia(semilla, tamano, cantidad_pois)
                inicio = time.perf_counter()
                resultado = resolver_csp(instancia, usar_heuristicas=heuristicas)
                computo.append((time.perf_counter() - inicio) * 1000.0)
                nodos.append(float(resultado.nodos_explorados))
                retrocesos.append(float(resultado.retrocesos))
                tiempos.append(resultado.tiempo_mision)
                costos.append(resultado.costo_ruta)
            nombre = "MRV-LCV-FC" if heuristicas else "Orden-fijo-FC"
            nm, nd = _resumen(nodos)
            bm, bd = _resumen(retrocesos)
            cm, cd = _resumen(costos)
            tm, td = _resumen(tiempos)
            sm, sd = _resumen(computo)
            resultados.append(
                (f"{tamano}/{nombre}", nm, nd, bm, bd, cm, cd, tm, td, sm, sd)
            )
    return resultados


def ejecutar_experimentos_csp_restringido(
    semillas: Sequence[int] = CSP_STRESS_SEEDS,
) -> List[CSPExperimentStats]:
    """Compara estrategias CSP en instancias pequeñas con restricciones ajustadas."""
    if not semillas:
        raise ValueError("Se requiere al menos una semilla.")

    metricas = {
        heuristicas: {
            "nodos": [],
            "retrocesos": [],
            "costos": [],
            "misiones": [],
            "computo": [],
        }
        for heuristicas in (False, True)
    }
    factibles = {False: 0, True: 0}
    for semilla in semillas:
        instancia = generar_instancia(semilla, tamano=8, num_pois=5)
        instancia.ventanas_comunicacion = CSP_STRESS_WINDOWS
        for heuristicas in (False, True):
            inicio = time.perf_counter()
            try:
                resultado = resolver_csp(
                    instancia,
                    capacidad_bateria=CSP_STRESS_BATTERY,
                    capacidad_memoria=MEMORY_CAPACITY,
                    limite_tiempo=CSP_STRESS_DEADLINE,
                    usar_heuristicas=heuristicas,
                )
            except RuntimeError as error:
                if str(error) != "El CSP no encontró una misión factible.":
                    raise
                continue
            valores = metricas[heuristicas]
            valores["nodos"].append(float(resultado.nodos_explorados))
            valores["retrocesos"].append(float(resultado.retrocesos))
            valores["costos"].append(resultado.costo_ruta)
            valores["misiones"].append(resultado.tiempo_mision)
            valores["computo"].append((time.perf_counter() - inicio) * 1000.0)
            factibles[heuristicas] += 1

    resultados = []
    for heuristicas in (False, True):
        valores = metricas[heuristicas]
        if not valores["nodos"]:
            raise RuntimeError("Ninguna instancia restringida resultó factible.")
        resumenes = [_resumen(valores[clave]) for clave in (
            "nodos",
            "retrocesos",
            "costos",
            "misiones",
            "computo",
        )]
        resultados.append(
            CSPExperimentStats(
                estrategia=(
                    "MRV/LCV+FC" if heuristicas else "Cronológico+FC"
                ),
                instancias=len(semillas),
                factibles=factibles[heuristicas],
                nodos_media=resumenes[0][0],
                nodos_desv=resumenes[0][1],
                retrocesos_media=resumenes[1][0],
                retrocesos_desv=resumenes[1][1],
                costo_media=resumenes[2][0],
                costo_desv=resumenes[2][1],
                mision_media=resumenes[3][0],
                mision_desv=resumenes[3][1],
                computo_media=resumenes[4][0],
                computo_desv=resumenes[4][1],
            )
        )
    return resultados


def imprimir_tabla_busqueda(
    resultados: Sequence[Tuple[str, str, float, float, float, float, float, float]]
) -> str:
    lineas = [
        "N | Algoritmo | Tiempo ms (media ± DE) | Expandidos (media ± DE) | Costo (media ± DE)",
        "--|-----------|------------------------|-------------------------|-------------------",
    ]
    for n, algoritmo, tm, td, em, ed, cm, cd in resultados:
        lineas.append(
            f"{n} | {algoritmo} | {tm:.4f} ± {td:.4f} | "
            f"{em:.2f} ± {ed:.2f} | {cm:.3f} ± {cd:.3f}"
        )
    return "\n".join(lineas)


def imprimir_tabla_metaheuristicas(
    resultados: Sequence[
        Tuple[str, float, float, float, float, float, float, float, float]
    ]
) -> str:
    lineas = [
        "N/algoritmo | Vecino cercano | Costo (media ± DE) | Mejora % (media ± DE) | Tiempo ms (media ± DE)",
        "------------|----------------|--------------------|----------------------|----------------------",
    ]
    for nombre, bm, bd, cm, cd, im, idesv, tm, td in resultados:
        lineas.append(
            f"{nombre} | {bm:.3f} ± {bd:.3f} | {cm:.3f} ± {cd:.3f} | "
            f"{im:.2f} ± {idesv:.2f} | {tm:.3f} ± {td:.3f}"
        )
    return "\n".join(lineas)


def imprimir_tabla_csp(
    resultados: Sequence[
        Tuple[str, float, float, float, float, float, float, float, float, float, float]
    ]
) -> str:
    lineas = [
        "N/configuración | Nodos (media ± DE) | Retrocesos (media ± DE) | Ruta (media ± DE) | Misión (media ± DE) | Cómputo ms (media ± DE)",
        "----------------|--------------------|--------------------------|------------------|---------------------|------------------------",
    ]
    for nombre, nm, nd, bm, bd, cm, cd, tm, td, sm, sd in resultados:
        lineas.append(
            f"{nombre} | {nm:.2f} ± {nd:.2f} | {bm:.2f} ± {bd:.2f} | "
            f"{cm:.3f} ± {cd:.3f} | {tm:.2f} ± {td:.2f} | {sm:.3f} ± {sd:.3f}"
        )
    return "\n".join(lineas)


#####SECCION 7: DEMOSTRACION Y LINEA DE COMANDOS
def imprimir_tabla_csp_restringido(
    resultados: Sequence[CSPExperimentStats],
) -> str:
    lineas = [
        "Estrategia | Factibles | Nodos probados (media ± DE) | Retrocesos (media ± DE) | "
        "Ruta (media ± DE) | Misión (media ± DE) | Cómputo ms (media ± DE)",
        "-----------|-----------|-----------------------------|-------------------------|"
        "-------------------|---------------------|------------------------",
    ]
    for resultado in resultados:
        lineas.append(
            f"{resultado.estrategia} | {resultado.factibles}/{resultado.instancias} | "
            f"{resultado.nodos_media:.2f} ± {resultado.nodos_desv:.2f} | "
            f"{resultado.retrocesos_media:.2f} ± {resultado.retrocesos_desv:.2f} | "
            f"{resultado.costo_media:.3f} ± {resultado.costo_desv:.3f} | "
            f"{resultado.mision_media:.2f} ± {resultado.mision_desv:.2f} | "
            f"{resultado.computo_media:.3f} ± {resultado.computo_desv:.3f}"
        )
    return "\n".join(lineas)


def demo() -> None:
    instancia = generar_instancia(seed=17, tamano=10, num_pois=4)
    print(
        f"Instancia {instancia.nombre}: base={instancia.base}, "
        f"comunicación={instancia.zona_comunicacion}, POI={instancia.puntos_interes}"
    )
    for algoritmo in SEARCH_ALGORITHMS:
        resultado = buscar_ruta(
            instancia, instancia.base, instancia.zona_comunicacion, algoritmo
        )
        print(
            f"{algoritmo}: ruta={resultado.camino}, costo={resultado.costo:.3f}, "
            f"expandidos={resultado.nodos_expandidos}"
        )
    metaheuristica = optimizar_orden(
        instancia,
        "simulated_annealing",
        seed=instancia.seed + 211,
        presupuesto=200,
    )
    print(
        f"Recocido simulado: orden={metaheuristica.orden}, "
        f"costo estimado={metaheuristica.costo:.3f}"
    )
    csp = resolver_csp(instancia, orden_preferido=metaheuristica.orden)
    print(
        f"CSP: orden={csp.orden}; comunicación después de posiciones "
        f"{csp.comunicar_despues_de}; tiempo={csp.tiempo_mision:.2f}"
    )
    for accion in csp.acciones:
        print(f"  - {accion}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Búsqueda, metaheurísticas y CSP del rover."
    )
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--experimentos-busqueda", action="store_true")
    parser.add_argument("--experimentos-metaheuristicas", action="store_true")
    parser.add_argument("--experimentos-csp", action="store_true")
    parser.add_argument("--tamanos", nargs="*", type=int, default=[8, 12, 16])
    parser.add_argument("--semillas", nargs="*", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--repeticiones-tiempo", type=int, default=3)
    parser.add_argument("--repeticiones-meta", type=int, default=8)
    parser.add_argument("--presupuesto", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tamanos = args.tamanos
    semillas = args.semillas
    if args.experimentos_busqueda:
        print(
            imprimir_tabla_busqueda(
                ejecutar_experimentos_busqueda(
                    tamanos, semillas, args.repeticiones_tiempo
                )
            )
        )
    if args.experimentos_metaheuristicas:
        print(
            imprimir_tabla_metaheuristicas(
                ejecutar_experimentos_metaheuristicas(
                    tamanos, semillas, args.presupuesto, args.repeticiones_meta
                )
            )
        )
    if args.experimentos_csp:
        print(imprimir_tabla_csp(ejecutar_experimentos_csp(tamanos, semillas)))
        print(
            "\nCSP con restricciones ajustadas "
            "(8 nodos, 5 POI, batería 9.0, plazo 25, ventanas de 6 unidades):"
        )
        print(
            imprimir_tabla_csp_restringido(
                ejecutar_experimentos_csp_restringido()
            )
        )
    if args.demo or not any(
        (
            args.experimentos_busqueda,
            args.experimentos_metaheuristicas,
            args.experimentos_csp,
        )
    ):
        demo()


if __name__ == "__main__":
    main()
