from collections import deque, defaultdict
from collections.abc import Iterable

import warnings


class Symbol(str):
    def __new__(cls, name):
        return str.__new__(cls, name)


class Node:
    def __init__(self, inputs, func, outputs, name=None):
        self.inputs = inputs
        self.func = func
        self.outputs = outputs
        self.name = name

    def __iter__(self):
        return [self.inputs, self.func, self.outputs, self.name]

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


# class Node:
#     def __init__(self, inputs, outputs, name=None):
#         self.inputs = inputs
#         self.func = self._func_error
#         self.outputs = outputs
#         self.name = name
#
#     @staticmethod
#     def _func_error(*args, **kwargs):
#         raise NotImplementedError("Function not registered")
#
#     def __call__(self, func, overwrite=False):
#         if self.func == self._func_error or overwrite:
#             self.func = func
#         else:
#             raise UserWarning("Function already registered")
#
#     def __iter__(self):
#         return [self.inputs, self.func, self.outputs, self.name]


class Graph:
    def __init__(self, *nodes):
        self._nodes_by_name = {}
        self._nodes_by_output = {}
        self._node_index = 0
        self._value_ref_count = defaultdict(int)

        self.add_nodes(*nodes)

    def add_node(self, inputs, func, outputs, name=None):
        inputs = self._ensure_string_list(inputs)
        outputs = self._ensure_string_list(outputs)

        if name is None:
            name = str(self._node_index)
            self._node_index += 1

        if name in self._nodes_by_name:
            warnings.warn(f"module '{name}' is already exists.", UserWarning)

        node = Node(inputs, func, outputs, name)
        # node = Node(inputs, outputs, name)(func)

        self._nodes_by_name[name] = node

        for input in inputs:
            self._value_ref_count[input] += 1

        for output in outputs:
            self._nodes_by_output[output] = node

    def add_nodes(self, *nodes):
        for node in nodes:
            self.add_node(node.inputs, node.func, node.outputs, node.name)

    def build_path(self, inputs, outputs):
        inputs = self._ensure_string_list(inputs)
        outputs = self._ensure_string_list(outputs)

        # Find required nodes
        stack = []
        satisfied = set(inputs)
        required = deque(outputs)

        while required:
            cur = required.popleft()
            if cur not in satisfied:
                satisfied.add(cur)
                node = self._nodes_by_output[cur]
                if node not in stack:
                    stack.append(node)
                    for input in node.inputs:
                        if input not in satisfied:
                            required.append(input)

        # Sort nodes
        computed = set(inputs)
        pending = []
        order = []

        while stack:
            node = stack.pop()

            for input in node.inputs:
                if input not in computed:
                    pending.append(node)
                    break
            else:
                order.append(node)
                computed.update(node.outputs)

            if not stack and pending:
                stack = list(reversed(pending))
                pending = []

        nodes = {node.name: node for node in order}

        return Path(inputs, nodes, outputs)

    @staticmethod
    def _ensure_string_list(arg):
        if isinstance(arg, str):
            arg = [arg]
        elif isinstance(arg, Iterable):
            for item in arg:
                if not isinstance(item, str):
                    raise ValueError("Item should be string")
        else:
            raise ValueError("input and output should be string or iterable of string")

        return arg


class Path:
    def __init__(self, inputs, nodes, outputs):
        self.inputs = inputs
        self.nodes = nodes
        self.outputs = outputs

    def __call__(self, *args, **kwargs):
        values = {}

        for name, value in zip(self.inputs, args):
            values[name] = value

        values.update(kwargs)

        name, node = None, None

        try:
            for name, node in self.nodes.items():
                # input_kwargs = {key: values[key] for key in input_keys}
                # output = module(**input_kwargs)
                input_args = [values[input] for input in node.inputs]
                output_values = node.func(*input_args)

                if len(node.outputs) == 1:
                    values[node.outputs[0]] = output_values
                else:
                    for key, value in zip(node.outputs, output_values):
                        values[key] = value

        # TODO: Error handling
        except BaseException as error:
            print(name, node.inputs, node.outputs, node.func)
            raise error

        if len(self.outputs) == 1:
            return values[self.outputs[0]]
        else:
            return [values[key] for key in self.outputs]

    def forward(self, *args, **kwargs):
        self(*args, **kwargs)
