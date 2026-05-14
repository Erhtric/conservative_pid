import itertools
import math
from abc import ABC, abstractmethod
from .io import CausalQuery, CausalExpression


class ResponseSignature(ABC):
    """
    Abstract base class for all response signatures.
    """

    def __init__(self, domains: dict[str, int]):
        self.domains: dict[str, int] = domains
        self.ordered_nodes: list[str] = []
        self.structure: dict[str, list[str]] = {}
        self.space: list[tuple[int]] = []

    @abstractmethod
    def _build_structure(self):
        """Must be implemented by subclasses to define the topological parent mapping."""
        pass

    def _generate_space(self):
        """Generates the Cartesian product of functional mappings based on the structure."""
        funcs_per_node = []
        for node in self.ordered_nodes:
            parents = self.structure[node]
            input_space_size = (
                math.prod([self.domains[p] for p in parents]) if parents else 1
            )
            node_funcs = list(
                itertools.product(range(self.domains[node]), repeat=input_space_size)
            )
            funcs_per_node.append(node_funcs)

        self.space = list(itertools.product(*funcs_per_node))

    @property
    def size(self):
        return len(self.space)

    def __str__(self):
        return f"ResponseSignature with {len(self.ordered_nodes)} nodes and {self.size:,} functions"

    def __repr__(self):
        return self.__str__()

    # ===============================================
    # Graphical Visalisation
    # ===============================================

    def build_canonical_pscm(self):
        """
        Constructs the canonical Partial SCM graph based on the structure.
        Each node has an edge from a single exogenous variable 'U' and edges from its
        parents.

        Returns:
            A networkx DiGraph representing the canonical PSCM.
        """
        import networkx as nx

        G = nx.DiGraph()
        node_positions = {
            node: (index, 0) for index, node in enumerate(self.ordered_nodes)
        }
        for node, parents in self.structure.items():
            G.add_node(node, pos=node_positions[node])
            child_index = self.ordered_nodes.index(node)
            for parent in parents:
                parent_index = self.ordered_nodes.index(parent)
                span = abs(child_index - parent_index)
                bend = min(0.25, 0.08 * span)
                G.add_edge(
                    parent,
                    node,
                    connectionstyle=f"arc3,rad={bend}",
                    span=span,
                )

        # Add exogenous variable
        G.add_node("U", pos=((len(self.ordered_nodes) - 1) / 2, -0.5))
        for node in self.ordered_nodes:
            span = abs(
                self.ordered_nodes.index(node) - (len(self.ordered_nodes) - 1) / 2
            )
            bend = min(0.25, 0.05 * span)
            G.add_edge("U", node, connectionstyle=f"arc3,rad={bend}", span=span)

        return G

    def canonical_edge_connectionstyles(self) -> list[str]:
        """Return edge connection styles in graph edge order for curved drawing."""
        graph = self.build_canonical_pscm()
        return [
            graph[u][v].get("connectionstyle", "arc3,rad=0.0") for u, v in graph.edges()
        ]

    @staticmethod
    def draw_canonical_pscm(signature_obj: ResponseSignature):
        import matplotlib.pyplot as plt
        import networkx as nx

        G = signature_obj.build_canonical_pscm()
        pos = nx.get_node_attributes(G, "pos")

        nx.draw_networkx_nodes(G, pos, node_size=400)
        nx.draw_networkx_labels(G, pos)

        for u, v, data in G.edges(data=True):
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(u, v)],
                arrows=True,
                arrowstyle="-|>",
                connectionstyle=data.get("connectionstyle", "arc3,rad=0.0"),
            )

        plt.axis("off")
        return plt


class TotalOrderSignature(ResponseSignature):
    """
    Constructs a complete canonical signature space based on a strict total order.
    """

    def __init__(self, domains: dict[str, int], total_order: list[str]):
        super().__init__(domains)
        self.ordered_nodes: list[str] = total_order
        self._build_structure()
        self._generate_space()

    def _build_structure(self):
        """In a total order, each node's parents are all preceding nodes in the order."""
        for i, node in enumerate(self.ordered_nodes):
            self.structure[node] = self.ordered_nodes[:i]


class PartialOrderSignature(ResponseSignature):
    """
    Constructs a reduced signature space enforcing parallel roots (interventions)
    and a targets ordered by domain size.
    """

    def __init__(self, domains: dict[str, int], query: CausalQuery | CausalExpression):
        super().__init__(domains)
        self.query: CausalQuery | CausalExpression = query
        self._build_structure()
        self._generate_space()

    def _build_structure(self):
        """Builds the structure based on the query's counterfactuals and evidence."""
        if isinstance(self.query, CausalQuery):
            roots_set: set[str] = set()
            for cf in self.query.counterfactuals:
                roots_set.update(cf.interventions.keys())
            # NOTE: observational evidence variables are also roots (?)
            roots_set.update(self.query.evidence.keys())

            roots = list(roots_set)
            print(f"Identified roots: {roots}")

            outcomes = list({cf.target_var for cf in self.query.counterfactuals})
            outcomes.sort(key=lambda v: self.domains[v])

            self.ordered_nodes = roots + outcomes
            self.structure = {r: [] for r in roots}

            current_parents = list(roots)
            for o in outcomes:
                self.structure[o] = list(current_parents)
                current_parents.append(o)

        elif isinstance(self.query, CausalExpression):
            target_vars = self.query.target_variables
            intervention_vars = self.query.intervention_variables

            roots = sorted(intervention_vars)
            outcomes = sorted(target_vars, key=lambda v: self.domains[v])
            self.ordered_nodes = roots + outcomes
            self.structure = {r: [] for r in roots}
            current_parents = list(roots)
            for o in outcomes:
                self.structure[o] = list(current_parents)
                current_parents.append(o)
