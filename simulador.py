#!/usr/bin/env python3
"""Visualizador local del rover y animador paso a paso de sus algoritmos."""

from __future__ import annotations

#####SECCION 1: IMPORTACIONES Y CONFIGURACION DE LA INTERFAZ
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Dict, Generator, List, Optional, Sequence, Tuple, Union

import rover

ANCHO_MAPA = 760
ALTO_MAPA = 620
MARGEN = 55
COLORES_TERRENO = {
    "llano": "#a7adb4",
    "roca": "#7f8c8d",
    "arena": "#d4ac6e",
    "duna": "#d98948",
}


#####SECCION 2: APLICACION GRAFICA DEL SIMULADOR
class SimuladorRover:
    """Interfaz Tkinter para generar mapas y observar cómputo en ejecución."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Rover autónomo — simulador de búsqueda, metaheurísticas y CSP")
        self.root.minsize(1000, 690)

        self.instancia: Optional[rover.RoverInstance] = None
        self.modo = tk.StringVar(value="Dijkstra")
        self.semilla = tk.StringVar(value="17")
        self.tamano = tk.StringVar(value="10")
        self.cantidad_pois = tk.StringVar(value="4")
        self.demora = tk.StringVar(value="120")
        self.estado = tk.StringVar(value="Genera un mapa y selecciona un algoritmo.")
        self.detalle = tk.StringVar(value="")
        self.recursos = tk.StringVar(value="")

        self.pasos: Optional[
            Generator[
                Union[rover.SearchEvent, rover.MetaheuristicEvent, rover.CSPEvent],
                None,
                Union[rover.SearchResult, rover.MetaheuristicResult, rover.CSPResult],
            ]
        ] = None
        self.tipo_animacion = ""
        self.en_ejecucion = False
        self.esperando = False
        self.eventos_expandidos: set[int] = set()
        self.frontera: set[int] = set()
        self.ruta_resaltada: List[Tuple[int, int]] = []
        self.nodo_actual: Optional[int] = None
        self.ruta_movimiento: List[int] = []
        self.plan_acciones: List[str] = []
        self.indice_plan = 0
        self.bateria = rover.BATTERY_CAPACITY
        self.memoria: set[int] = set()
        self.analizadas: set[int] = set()
        self.transmitidas: set[int] = set()
        self.tiempo_mision = 0.0
        self.variable_asignada: Dict[int, int] = {}
        self.orden_meta: List[int] = []

        self._construir_interfaz()
        self.generar_mapa()

    def _construir_interfaz(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        controles = ttk.Frame(self.root, padding=(10, 8))
        controles.grid(row=0, column=0, sticky="ew")
        controles.columnconfigure(10, weight=1)

        ttk.Label(controles, text="Algoritmo").grid(row=0, column=0, padx=(0, 4))
        algoritmo = ttk.Combobox(
            controles,
            textvariable=self.modo,
            values=(
                "BFS",
                "Dijkstra",
                "A* h=0",
                "A* euclídea",
                "Recocido simulado",
                "Ascenso de colinas",
                "CSP",
            ),
            state="readonly",
            width=21,
        )
        algoritmo.grid(row=0, column=1, padx=(0, 10))

        for columna, etiqueta, variable, ancho in (
            (2, "Semilla", self.semilla, 6),
            (4, "Nodos", self.tamano, 5),
            (6, "POI", self.cantidad_pois, 4),
            (8, "Demora ms", self.demora, 6),
        ):
            ttk.Label(controles, text=etiqueta).grid(
                row=0, column=columna, padx=(0, 3)
            )
            ttk.Entry(controles, textvariable=variable, width=ancho).grid(
                row=0, column=columna + 1, padx=(0, 8)
            )

        ttk.Button(controles, text="Generar mapa", command=self.generar_mapa).grid(
            row=0, column=10, padx=4
        )
        ttk.Button(controles, text="Iniciar", command=self.iniciar).grid(
            row=0, column=11, padx=4
        )
        ttk.Button(controles, text="Paso", command=self.paso).grid(
            row=0, column=12, padx=4
        )
        self.boton_ejecucion = ttk.Button(
            controles, text="Reproducir", command=self.alternar_ejecucion
        )
        self.boton_ejecucion.grid(row=0, column=13, padx=4)

        cuerpo = ttk.Frame(self.root, padding=(10, 2, 10, 8))
        cuerpo.grid(row=1, column=0, sticky="nsew")
        cuerpo.columnconfigure(0, weight=1)
        cuerpo.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            cuerpo,
            width=ANCHO_MAPA,
            height=ALTO_MAPA,
            background="#f7f8fa",
            highlightthickness=1,
            highlightbackground="#c7cbd1",
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")

        panel = ttk.Frame(cuerpo, padding=(12, 4), width=290)
        panel.grid(row=0, column=1, sticky="nsew")
        panel.grid_propagate(False)
        ttk.Label(panel, text="Ejecución", font=("Segoe UI", 13, "bold")).pack(
            anchor="w", pady=(5, 10)
        )
        ttk.Label(
            panel,
            textvariable=self.estado,
            wraplength=265,
            justify="left",
        ).pack(anchor="w", pady=(0, 12))
        ttk.Separator(panel).pack(fill="x", pady=6)
        ttk.Label(panel, text="Paso actual", font=("Segoe UI", 10, "bold")).pack(
            anchor="w", pady=(4, 4)
        )
        ttk.Label(
            panel,
            textvariable=self.detalle,
            wraplength=265,
            justify="left",
        ).pack(anchor="w", pady=(0, 12))
        ttk.Label(panel, text="Recursos y progreso", font=("Segoe UI", 10, "bold")).pack(
            anchor="w", pady=(4, 4)
        )
        ttk.Label(
            panel,
            textvariable=self.recursos,
            wraplength=265,
            justify="left",
        ).pack(anchor="w", pady=(0, 12))
        ttk.Separator(panel).pack(fill="x", pady=6)
        leyenda = (
            "Leyenda\n"
            "■ Base    ■ Comunicación    ■ POI\n"
            "■ Frontera    ■ Expandido    ■ Ruta\n\n"
            "Terrenos: llano, roca, arena y duna.\n"
            "Usa Paso para avanzar manualmente o Reproducir para animar."
        )
        ttk.Label(panel, text=leyenda, justify="left", wraplength=265).pack(
            anchor="w", pady=8
        )
        ttk.Label(
            self.root,
            text="Proyecto 1 · El mapa y las ejecuciones se generan localmente; no requiere paquetes externos.",
            anchor="w",
            padding=(12, 0, 12, 8),
        ).grid(row=2, column=0, sticky="ew")

    #####SECCION 3: GENERACION Y DIBUJO DEL MAPA
    def generar_mapa(self) -> None:
        try:
            seed = int(self.semilla.get())
            n = int(self.tamano.get())
            num_pois = int(self.cantidad_pois.get())
            demora = int(self.demora.get())
            if demora < 1:
                raise ValueError("La demora debe ser de al menos 1 ms.")
            if num_pois > n - 2:
                raise ValueError("El mapa necesita un nodo para base y otro para comunicación.")
            self.instancia = rover.generar_instancia(seed, n, num_pois)
        except ValueError as error:
            messagebox.showerror("Parámetros inválidos", str(error))
            return

        self._detener()
        self.eventos_expandidos.clear()
        self.frontera.clear()
        self.ruta_resaltada.clear()
        self.nodo_actual = self.instancia.base
        self.variable_asignada.clear()
        self.plan_acciones.clear()
        self.memoria.clear()
        self.analizadas.clear()
        self.transmitidas.clear()
        self.bateria = rover.BATTERY_CAPACITY
        self.tiempo_mision = 0.0
        self.estado.set(
            f"{self.instancia.nombre}: base {self.instancia.base}, "
            f"comunicación {self.instancia.zona_comunicacion}, "
            f"{len(self.instancia.puntos_interes)} POI."
        )
        self.detalle.set("Mapa generado con una semilla reproducible.")
        self._actualizar_recursos()
        self._dibujar_mapa()

    def _coordenada(self, nodo: int) -> Tuple[float, float]:
        assert self.instancia is not None
        x, y = self.instancia.posiciones[nodo]
        ancho = max(self.canvas.winfo_width(), ANCHO_MAPA)
        alto = max(self.canvas.winfo_height(), ALTO_MAPA)
        return (
            MARGEN + x * (ancho - 2 * MARGEN),
            MARGEN + y * (alto - 2 * MARGEN),
        )

    def _dibujar_mapa(self) -> None:
        if self.instancia is None:
            return
        self.canvas.delete("all")
        coords = {n: self._coordenada(n) for n in range(self.instancia.n)}

        for origen in range(self.instancia.n):
            for destino, costo, terreno in self.instancia.vecinos(origen):
                if origen >= destino:
                    continue
                x1, y1 = coords[origen]
                x2, y2 = coords[destino]
                color = COLORES_TERRENO.get(terreno, "#a7adb4")
                if (origen, destino) in self.ruta_resaltada or (
                    destino,
                    origen,
                ) in self.ruta_resaltada:
                    color = "#2563eb"
                self.canvas.create_line(
                    x1,
                    y1,
                    x2,
                    y2,
                    fill=color,
                    width=3 if color == "#2563eb" else 1.6,
                )
                xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
                self.canvas.create_rectangle(
                    xm - 16, ym - 10, xm + 16, ym + 10, fill="#f7f8fa", outline=""
                )
                self.canvas.create_text(
                    xm,
                    ym,
                    text=f"{terreno[:2]} {costo:g}",
                    fill="#343a40",
                    font=("Segoe UI", 8),
                )

        for nodo, (x, y) in coords.items():
            if nodo == self.instancia.base:
                color, etiqueta = "#273746", "BASE"
            elif nodo == self.instancia.zona_comunicacion:
                color, etiqueta = "#2874a6", "COM"
            elif nodo in self.instancia.puntos_interes:
                color, etiqueta = "#d68910", "POI"
            elif nodo in self.eventos_expandidos:
                color, etiqueta = "#55a868", "EXP"
            elif nodo in self.frontera:
                color, etiqueta = "#f4d03f", "FRON"
            else:
                color, etiqueta = "#aeb6bf", "PASO"
            if nodo in self.variable_asignada.values():
                color = "#8e44ad"
            self.canvas.create_oval(
                x - 18,
                y - 18,
                x + 18,
                y + 18,
                fill=color,
                outline="#ffffff",
                width=2,
            )
            self.canvas.create_text(
                x,
                y,
                text=str(nodo),
                fill="white",
                font=("Segoe UI", 9, "bold"),
            )
            self.canvas.create_text(
                x,
                y + 29,
                text=etiqueta,
                fill="#343a40",
                font=("Segoe UI", 8),
            )
            if nodo in self.memoria:
                self.canvas.create_text(
                    x + 21,
                    y - 19,
                    text="M",
                    fill="#7d3c98",
                    font=("Segoe UI", 9, "bold"),
                )
            if nodo == self.nodo_actual:
                self.canvas.create_oval(
                    x - 7,
                    y - 7,
                    x + 7,
                    y + 7,
                    fill="#e74c3c",
                    outline="white",
                    width=1,
                )

    #####SECCION 4: CONTROL DE ALGORITMOS Y ANIMACION
    def iniciar(self) -> None:
        if self.instancia is None:
            return
        self._detener()
        self.eventos_expandidos.clear()
        self.frontera.clear()
        self.ruta_resaltada.clear()
        self.variable_asignada.clear()
        self.memoria.clear()
        self.analizadas.clear()
        self.transmitidas.clear()
        self.bateria = rover.BATTERY_CAPACITY
        self.tiempo_mision = 0.0
        self.nodo_actual = self.instancia.base
        self.ruta_movimiento = []
        self.indice_plan = 0
        self.plan_acciones = []
        self._actualizar_recursos()
        self._dibujar_mapa()

        modo = self.modo.get()
        try:
            if modo in {"BFS", "Dijkstra", "A* h=0", "A* euclídea"}:
                algoritmo = {
                    "BFS": "bfs",
                    "Dijkstra": "dijkstra",
                    "A* h=0": "astar_zero",
                    "A* euclídea": "astar_euclidean",
                }[modo]
                self.tipo_animacion = "busqueda"
                self.pasos = rover.trazar_busqueda(
                    self.instancia,
                    self.instancia.base,
                    self.instancia.zona_comunicacion,
                    algoritmo,
                )
            elif modo in {"Recocido simulado", "Ascenso de colinas"}:
                self.tipo_animacion = "metaheuristica"
                self.pasos = rover.trazar_metaheuristica(
                    self.instancia,
                    "simulated_annealing"
                    if modo == "Recocido simulado"
                    else "hill_climbing",
                    seed=self.instancia.seed + 101,
                    presupuesto=200,
                )
            else:
                self.tipo_animacion = "csp"
                orden_preferido = rover.orden_visita_vecino_mas_cercano(
                    self.instancia
                )
                self.pasos = rover.trazar_resolucion_csp(
                    self.instancia,
                    orden_preferido=orden_preferido,
                )
            self.estado.set(f"Ejecutando {modo}. Paso a paso con «Paso» o automáticamente.")
            self.detalle.set("Inicializando la frontera de búsqueda.")
            self.esperando = True
            self.boton_ejecucion.configure(text="Reproducir")
            self._dibujar_mapa()
        except (ValueError, RuntimeError) as error:
            messagebox.showerror("No se pudo iniciar", str(error))

    def paso(self) -> None:
        if self.esperando:
            self._paso_algoritmo()
        elif self.ruta_movimiento:
            self._mover_siguiente()
        elif self.plan_acciones and self.indice_plan < len(self.plan_acciones):
            self._ejecutar_accion_plan()

    def alternar_ejecucion(self) -> None:
        if not self.esperando and not self.ruta_movimiento and not (
            self.plan_acciones and self.indice_plan < len(self.plan_acciones)
        ):
            self.iniciar()
            return
        self.en_ejecucion = not self.en_ejecucion
        self.boton_ejecucion.configure(
            text="Pausar" if self.en_ejecucion else "Reproducir"
        )
        if self.en_ejecucion:
            self._programar_siguiente()

    def _programar_siguiente(self) -> None:
        if not self.en_ejecucion:
            return
        try:
            demora = max(1, int(self.demora.get()))
        except ValueError:
            demora = 120
        self.root.after(demora, self._paso_programado)

    def _paso_programado(self) -> None:
        if self.esperando:
            self._paso_algoritmo()
        elif self.ruta_movimiento:
            self._mover_siguiente()
        elif self.plan_acciones and self.indice_plan < len(self.plan_acciones):
            self._ejecutar_accion_plan()
        else:
            self.en_ejecucion = False
            self.boton_ejecucion.configure(text="Reproducir")
            return
        self._programar_siguiente()

    def _paso_algoritmo(self) -> None:
        if self.pasos is None:
            return
        try:
            evento = next(self.pasos)
        except StopIteration as resultado:
            self.esperando = False
            self._finalizar_algoritmo(resultado.value)
            return
        except (ValueError, RuntimeError) as error:
            self._detener()
            messagebox.showerror("Ejecución fallida", str(error))
            return

        if isinstance(evento, rover.SearchEvent):
            self.eventos_expandidos = set(evento.expandidos)
            self.frontera = set(evento.frontera)
            self.detalle.set(
                f"Expansión {evento.cantidad}: nodo {evento.nodo}; "
                f"frontera con {len(evento.frontera)} estados."
            )
            self._dibujar_mapa()
        elif isinstance(evento, rover.MetaheuristicEvent):
            self.orden_meta = list(evento.mejor_orden)
            self.ruta_resaltada = self._aristas_orden(self.orden_meta)
            self.detalle.set(
                f"Evaluación {evento.evaluacion}/200\n"
                f"Costo actual: {evento.costo_actual:.2f}\n"
                f"Mejor costo: {evento.mejor_costo:.2f}\n"
                f"Mejor orden: {list(evento.mejor_orden)}"
            )
            self._dibujar_mapa()
        elif isinstance(evento, rover.CSPEvent):
            self.variable_asignada = {
                posicion: poi for posicion, poi, _, _ in evento.asignados
            }
            self.detalle.set(
                f"{evento.tipo.upper()}: {evento.detalle}\n"
                f"Asignaciones: {len(evento.asignados)}; "
                f"valores en dominios: {evento.dominios_restantes}."
            )
            self._dibujar_mapa()

    #####SECCION 5: RESULTADOS Y EJECUCION DEL PLAN
    def _finalizar_algoritmo(self, resultado: object) -> None:
        if isinstance(resultado, rover.SearchResult):
            self.estado.set(
                f"{resultado.algoritmo}: costo {resultado.costo:.2f}; "
                f"{resultado.nodos_expandidos} nodos expandidos."
            )
            self.detalle.set(
                f"Ruta óptima/devuelta: {resultado.camino}\n"
                f"Tiempo de búsqueda: {resultado.tiempo_ms:.4f} ms."
            )
            self.ruta_resaltada = list(zip(resultado.camino, resultado.camino[1:]))
            self.ruta_movimiento = resultado.camino[1:]
            self.indice_plan = 0
            self._dibujar_mapa()
        elif isinstance(resultado, rover.MetaheuristicResult):
            self.orden_meta = resultado.orden
            self.ruta_resaltada = self._aristas_orden(resultado.orden)
            self.ruta_movimiento = self._nodos_orden(resultado.orden)
            self.estado.set(
                f"{resultado.algoritmo}: costo {resultado.costo:.2f}; "
                f"{resultado.evaluaciones} evaluaciones."
            )
            self.detalle.set(
                f"Mejor orden: {resultado.orden}\n"
                f"Tiempo: {resultado.tiempo_ms:.2f} ms.\n"
                "El rover recorrerá el orden resaltado."
            )
            self._dibujar_mapa()
        elif isinstance(resultado, rover.CSPResult):
            self.plan_acciones = resultado.acciones
            self.indice_plan = 0
            self.estado.set(
                f"CSP factible: costo {resultado.costo_ruta:.2f}, "
                f"plazo {resultado.tiempo_mision:.2f}; "
                f"{resultado.nodos_explorados} valores probados."
            )
            self.detalle.set(
                f"Orden: {resultado.orden}\n"
                f"Comunicación después de posiciones "
                f"{resultado.comunicar_despues_de}.\n"
                "Reproduciendo las acciones y actualizando recursos."
            )
            if self.plan_acciones:
                self._ejecutar_accion_plan()
        self._actualizar_recursos()

    def _ejecutar_accion_plan(self) -> None:
        if self.instancia is None or self.indice_plan >= len(self.plan_acciones):
            self.estado.set("Misión terminada.")
            return
        accion = self.plan_acciones[self.indice_plan]
        self.indice_plan += 1
        partes = accion.split()
        tipo = partes[0]
        if tipo == "move":
            origen, destino = map(int, partes[1:])
            costo = next(
                c for vecino, c, _ in self.instancia.vecinos(origen) if vecino == destino
            )
            self.nodo_actual = destino
            self.bateria -= rover._energia_arista(costo)
            self.tiempo_mision += costo
        elif tipo == "take":
            self.memoria.add(int(partes[1]))
            self.tiempo_mision += rover.TAKE_DURATION
        elif tipo == "analyze":
            self.analizadas.add(int(partes[1]))
            self.tiempo_mision += rover.ANALYZE_DURATION
        elif tipo == "transmit":
            poi = int(partes[1])
            self.memoria.discard(poi)
            self.transmitidas.add(poi)
            self.tiempo_mision += rover.TRANSMIT_DURATION
        elif tipo == "wait":
            self.tiempo_mision += float(partes[1])
        elif tipo == "recharge":
            self.bateria = rover.BATTERY_CAPACITY
            self.tiempo_mision += rover.RECHARGE_DURATION

        self.detalle.set(
            f"Acción {self.indice_plan}/{len(self.plan_acciones)}: {accion}"
        )
        self._actualizar_recursos()
        self._dibujar_mapa()
        if self.indice_plan == len(self.plan_acciones):
            self.estado.set(
                f"Misión completa: {len(self.transmitidas)} muestras transmitidas."
            )

    def _mover_siguiente(self) -> None:
        if not self.ruta_movimiento:
            self.estado.set("Recorrido terminado.")
            return
        siguiente = self.ruta_movimiento.pop(0)
        self.nodo_actual = siguiente
        self.detalle.set(f"El rover avanza al nodo {siguiente}.")
        self._actualizar_recursos()
        self._dibujar_mapa()

    #####SECCION 6: RUTAS, RECURSOS Y UTILIDADES DE CONTROL
    def _aristas_orden(self, orden: Sequence[int]) -> List[Tuple[int, int]]:
        if self.instancia is None:
            return []
        aristas: List[Tuple[int, int]] = []
        origen = self.instancia.base
        destinos = list(orden)
        destinos.append(self.instancia.zona_comunicacion)
        for destino in destinos:
            ruta = rover.buscar_ruta(self.instancia, origen, destino, "dijkstra").camino
            aristas.extend(zip(ruta, ruta[1:]))
            origen = destino
        return aristas

    def _nodos_orden(self, orden: Sequence[int]) -> List[int]:
        if self.instancia is None:
            return []
        nodos: List[int] = []
        origen = self.instancia.base
        destinos = list(orden) + [self.instancia.zona_comunicacion]
        for destino in destinos:
            ruta = rover.buscar_ruta(self.instancia, origen, destino, "dijkstra").camino
            nodos.extend(ruta[1:])
            origen = destino
        return nodos

    def _actualizar_recursos(self) -> None:
        self.recursos.set(
            f"Ubicación: {self.nodo_actual}\n"
            f"Batería: {max(0, self.bateria) / rover.ENERGY_SCALE:.1f} / "
            f"{rover.BATTERY_CAPACITY / rover.ENERGY_SCALE:.1f}\n"
            f"Memoria ocupada: {len(self.memoria)} / {rover.MEMORY_CAPACITY}\n"
            f"POI analizados: {len(self.analizadas)}\n"
            f"Muestras transmitidas: {len(self.transmitidas)}\n"
            f"Tiempo de misión: {self.tiempo_mision:.1f}"
        )

    def _detener(self) -> None:
        self.en_ejecucion = False
        self.esperando = False
        self.pasos = None
        self.boton_ejecucion.configure(text="Reproducir")


#####SECCION 7: PUNTO DE ENTRADA
def main() -> None:
    root = tk.Tk()
    SimuladorRover(root)
    root.mainloop()


if __name__ == "__main__":
    main()
