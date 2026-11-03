import unittest

import rover


class RoverTests(unittest.TestCase):
    def test_instance_generation_is_reproducible_and_connected(self):
        first = rover.generar_instancia(29, tamano=12, num_pois=5)
        second = rover.generar_instancia(29, tamano=12, num_pois=5)
        self.assertEqual(first.posiciones, second.posiciones)
        self.assertEqual(first.grafo, second.grafo)
        self.assertEqual(first.puntos_interes, second.puntos_interes)
        self.assertEqual(first.ventanas_comunicacion, second.ventanas_comunicacion)

        reached = {first.base}
        pending = [first.base]
        while pending:
            for neighbor, _, _ in first.vecinos(pending.pop()):
                if neighbor not in reached:
                    reached.add(neighbor)
                    pending.append(neighbor)
        self.assertEqual(reached, set(range(first.n)))

    def test_weighted_search_and_heuristics_return_optimal_cost(self):
        instance = rover.generar_instancia(17, tamano=10, num_pois=3)
        dijkstra = rover.buscar_ruta(
            instance, instance.base, instance.zona_comunicacion, "dijkstra"
        )
        zero_heuristic = rover.buscar_ruta(
            instance, instance.base, instance.zona_comunicacion, "astar_zero"
        )
        euclidean = rover.buscar_ruta(
            instance, instance.base, instance.zona_comunicacion, "astar_euclidean"
        )
        self.assertAlmostEqual(dijkstra.costo, zero_heuristic.costo)
        self.assertAlmostEqual(dijkstra.costo, euclidean.costo)
        self.assertEqual(euclidean.camino[0], instance.base)
        self.assertEqual(euclidean.camino[-1], instance.zona_comunicacion)

    def test_bfs_minimizes_hops_not_weighted_cost(self):
        instance = rover.RoverInstance(
            nombre="weighted-route-test",
            n=4,
            base=0,
            zona_comunicacion=3,
            puntos_interes=[2],
            posiciones={
                0: (0.0, 0.0),
                1: (0.0, 0.5),
                2: (0.5, 0.5),
                3: (1.0, 0.0),
            },
            grafo={
                0: [(1, 1.0, "llano"), (3, 3.1, "duna")],
                1: [(0, 1.0, "llano"), (3, 1.0, "llano")],
                2: [],
                3: [(0, 3.1, "duna"), (1, 1.0, "llano")],
            },
        )
        bfs = rover.buscar_ruta(instance, 0, 3, "bfs")
        dijkstra = rover.buscar_ruta(instance, 0, 3, "dijkstra")
        self.assertEqual(bfs.camino, [0, 3])
        self.assertEqual(dijkstra.camino, [0, 1, 3])
        self.assertGreater(bfs.costo, dijkstra.costo)

    def test_search_generator_exposes_each_expansion(self):
        instance = rover.generar_instancia(4, tamano=8, num_pois=2)
        events = []
        steps = rover.trazar_busqueda(
            instance, instance.base, instance.zona_comunicacion, "astar_euclidean"
        )
        while True:
            try:
                events.append(next(steps))
            except StopIteration as result:
                route = result.value
                break
        self.assertEqual(len(events), route.nodos_expandidos)
        self.assertEqual(events[-1].nodo, instance.zona_comunicacion)

    def test_metaheuristics_use_equal_evaluation_budget(self):
        instance = rover.generar_instancia(8, tamano=8, num_pois=4)
        for algorithm in ("hill_climbing", "simulated_annealing"):
            result = rover.optimizar_orden(
                instance, algorithm, seed=3, presupuesto=25
            )
            self.assertEqual(result.evaluaciones, 25)
            self.assertEqual(set(result.orden), set(instance.puntos_interes))
            repeated = rover.optimizar_orden(
                instance, algorithm, seed=3, presupuesto=25
            )
            self.assertEqual(result.orden, repeated.orden)
            self.assertEqual(result.costo, repeated.costo)

    def test_csp_plan_obeys_precedence_resources_windows_and_deadline(self):
        instance = rover.generar_instancia(17, tamano=10, num_pois=4)
        result = rover.resolver_csp(instance, capacidad_memoria=1)
        self._assert_plan_is_feasible(
            instance,
            result.acciones,
            capacidad_memoria=1,
            tiempo_limite=rover.MISSION_TIME_LIMIT,
        )
        self.assertEqual(set(result.orden), set(instance.puntos_interes))
        self.assertEqual(
            result.comunicar_despues_de, list(range(len(instance.puntos_interes)))
        )

    def test_csp_hard_instance_finds_feasible_plans_after_backtracking(self):
        instance = rover.generar_instancia(0, tamano=8, num_pois=5)
        instance.ventanas_comunicacion = rover.CSP_STRESS_WINDOWS
        for heuristicas in (False, True):
            result = rover.resolver_csp(
                instance,
                capacidad_bateria=rover.CSP_STRESS_BATTERY,
                capacidad_memoria=rover.MEMORY_CAPACITY,
                limite_tiempo=rover.CSP_STRESS_DEADLINE,
                usar_heuristicas=heuristicas,
            )
            self.assertGreater(result.retrocesos, 0)
            self._assert_plan_is_feasible(
                instance,
                result.acciones,
                capacidad_memoria=rover.MEMORY_CAPACITY,
                tiempo_limite=rover.CSP_STRESS_DEADLINE,
                capacidad_bateria=rover.CSP_STRESS_BATTERY,
            )

    def test_csp_rejects_instances_above_documented_poi_limit(self):
        instance = rover.generar_instancia(1, tamano=15, num_pois=13)
        with self.assertRaisesRegex(ValueError, "máximo 12"):
            rover.resolver_csp(instance)

    def test_csp_explicitly_fails_when_no_communication_window_exists(self):
        instance = rover.generar_instancia(5, tamano=8, num_pois=2)
        instance.ventanas_comunicacion = ()
        with self.assertRaisesRegex(RuntimeError, "no encontró"):
            rover.resolver_csp(instance)

    def test_csp_generator_exposes_assignments_and_propagation(self):
        instance = rover.generar_instancia(9, tamano=8, num_pois=3)
        events = []
        steps = rover.trazar_resolucion_csp(instance)
        while True:
            try:
                events.append(next(steps))
            except StopIteration as result:
                solution = result.value
                break
        self.assertTrue(any(event.tipo == "asignacion" for event in events))
        self.assertEqual(set(solution.orden), set(instance.puntos_interes))
        self.assertGreater(solution.nodos_explorados, 0)

    def test_csp_accepts_metaheuristic_order_as_a_feasibility_preference(self):
        instance = rover.generar_instancia(19, tamano=8, num_pois=4)
        preference = rover.optimizar_orden(
            instance, "simulated_annealing", seed=3, presupuesto=20
        ).orden
        result = rover.resolver_csp(instance, orden_preferido=preference)
        self.assertEqual(set(result.orden), set(instance.puntos_interes))
        with self.assertRaisesRegex(ValueError, "permutación exacta"):
            rover.resolver_csp(instance, orden_preferido=[preference[0]])

    def _assert_plan_is_feasible(
        self,
        instance,
        actions,
        capacidad_memoria,
        tiempo_limite,
        capacidad_bateria=rover.BATTERY_CAPACITY,
    ):
        location = instance.base
        battery = capacidad_bateria
        memory = set()
        collected = set()
        analyzed = set()
        transmitted = set()
        time_now = 0.0
        for action in actions:
            parts = action.split()
            kind = parts[0]
            if kind == "move":
                origin, destination = map(int, parts[1:])
                self.assertEqual(location, origin)
                edge_cost = next(
                    cost
                    for neighbor, cost, _ in instance.vecinos(origin)
                    if neighbor == destination
                )
                battery -= rover._energia_arista(edge_cost)
                self.assertGreaterEqual(battery, 0)
                time_now += edge_cost
                location = destination
            elif kind == "take":
                poi = int(parts[1])
                self.assertEqual(location, poi)
                self.assertIn(poi, instance.puntos_interes)
                self.assertNotIn(poi, collected)
                self.assertLess(len(memory), capacidad_memoria)
                collected.add(poi)
                memory.add(poi)
                time_now += rover.TAKE_DURATION
            elif kind == "analyze":
                poi = int(parts[1])
                self.assertIn(poi, collected)
                self.assertNotIn(poi, analyzed)
                analyzed.add(poi)
                time_now += rover.ANALYZE_DURATION
            elif kind == "wait":
                self.assertEqual(location, instance.zona_comunicacion)
                time_now += float(parts[1])
            elif kind == "transmit":
                poi, zone = map(int, parts[1:])
                self.assertEqual(location, zone)
                self.assertEqual(location, instance.zona_comunicacion)
                self.assertIn(poi, analyzed)
                self.assertIn(poi, memory)
                window = next(
                    (
                        (start, end)
                        for start, end in instance.ventanas_comunicacion
                        if start <= time_now + 1e-8
                        and time_now + rover.TRANSMIT_DURATION <= end + 1e-8
                    ),
                    None,
                )
                self.assertIsNotNone(window, f"Transmisión fuera de ventana: {action}")
                memory.remove(poi)
                transmitted.add(poi)
                time_now += rover.TRANSMIT_DURATION
            elif kind == "recharge":
                self.assertIn(location, {instance.base, instance.zona_comunicacion})
                self.assertLess(battery, capacidad_bateria)
                battery = capacidad_bateria
                time_now += rover.RECHARGE_DURATION
            else:
                self.fail(f"Acción desconocida: {action}")

        self.assertEqual(transmitted, set(instance.puntos_interes))
        self.assertFalse(memory)
        self.assertEqual(location, instance.zona_comunicacion)
        self.assertLessEqual(time_now, tiempo_limite)


if __name__ == "__main__":
    unittest.main()
