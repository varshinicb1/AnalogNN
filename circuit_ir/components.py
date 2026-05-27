class Component:
    def __init__(self, name: str, nodes: list, value=None):
        self.name = name
        self.nodes = nodes
        self.value = value

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name}, nodes={self.nodes}, value={self.value})"


class Resistor(Component):
    def __init__(self, name: str, node1: str, node2: str, value: float):
        super().__init__(name, [node1, node2], value)


class Capacitor(Component):
    def __init__(self, name: str, node1: str, node2: str, value: float):
        super().__init__(name, [node1, node2], value)


class OpAmp(Component):
    def __init__(self, name: str, pos_node: str, neg_node: str, out_node: str, ref_node: str = "0", open_loop_gain: float = 1e5):
        super().__init__(name, [pos_node, neg_node, out_node, ref_node], open_loop_gain)
        self.open_loop_gain = open_loop_gain


class VoltageSource(Component):
    def __init__(self, name: str, node1: str, node2: str, value: float, source_type: str = "DC"):
        """
        source_type can be 'DC', 'AC', or transient options like 'PULSE', 'SIN'
        """
        super().__init__(name, [node1, node2], value)
        self.source_type = source_type


class CurrentSource(Component):
    def __init__(self, name: str, node1: str, node2: str, value: float):
        super().__init__(name, [node1, node2], value)
