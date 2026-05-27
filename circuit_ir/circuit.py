from circuit_ir.components import Component

class Circuit:
    def __init__(self, name: str = "AnalogNeuralSubsystem"):
        self.name = name
        self.components = []
        self.nodes = set()

    def add_component(self, component: Component):
        self.components.append(component)
        for n in component.nodes:
            self.nodes.add(str(n))
        return component

    def get_components_of_type(self, comp_class):
        return [c for c in self.components if isinstance(c, comp_class)]

    def get_all_nodes(self) -> list:
        return sorted(list(self.nodes))

    def clear(self):
        self.components.clear()
        self.nodes.clear()

    def __repr__(self):
        return f"Circuit(name={self.name}, components={len(self.components)}, nodes={len(self.nodes)})"
