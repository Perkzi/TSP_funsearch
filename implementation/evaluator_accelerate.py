# Implemented by RZ.
# This file aims to accelerate the original evaluate logic using 'numba' package.
# You should install numba package in your Python environment or the later evaluation will fail.

import ast


def add_numba_decorator(
        program: str,
        function_to_evolve: str,
) -> str:
    """
    This function aims to accelerate the evaluation of the searched code. This is achieved by decorating '@numba.jit()'
    to the function_to_evolve. However, it should be noted that not all numpy functions support numba acceleration:
    such as np.piecewise(). So use this function wisely. Hahaha!
    这个函数 add_numba_decorator 的作用是为指定的函数添加 @numba.jit 装饰器，以加速代码的执行。
    具体来说，它会在代码中插入 import numba 语句，并为指定的函数添加 @numba.jit(nopython=True) 装饰器。

    Example input program:
        def func(a: np.ndarray):
            return a * 2
    Example output program
        import numba

        numba.jit()
        def func(a: np.ndarray):
            return a * 2
    """
    # parse to syntax tree
    tree = ast.parse(program) # 将代码字符串解析为语法树。语法树是一种表示代码结构的树状数据结构，每个节点代表代码中的一个元素。

    # check if 'import numba' already exists
    numba_imported = False
    for node in tree.body:
        if isinstance(node, ast.Import) and any(alias.name == 'numba' for alias in node.names):
            numba_imported = True
            break

    # add 'import numba' to the top of the program
    if not numba_imported:
        import_node = ast.Import(names=[ast.alias(name='numba', asname=None)])
        tree.body.insert(0, import_node)

    # traverse the tree, and find the function_to_run
    for node in ast.walk(tree):
        # 找到名称为 function_to_evolve 的函数
        if isinstance(node, ast.FunctionDef) and node.name == function_to_evolve:
            # the @numba.jit() decorator instance
            # ast.Call：创建一个表示函数调用的语法树节点。在这个例子中，它用于创建 @numba.jit(nopython=True) 装饰器节点
            decorator = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='numba', ctx=ast.Load()),
                    attr='jit',
                    ctx=ast.Load()
                ),
                args=[],  # args do not have argument name
                keywords=[ast.keyword(arg='nopython', value=ast.NameConstant(value=True))]
                # keywords have argument name
            )
            # add the decorator to the decorator_list of the node
            node.decorator_list.append(decorator)

    # turn the tree to string and return将语法树转换回代码字符串
    modified_program = ast.unparse(tree)
    return modified_program


if __name__ == '__main__':
    code = '''
import numpy as np
import numba

def func1():
    return 3

def func():
    return 5
    '''
    res = add_numba_decorator(code, 'func')
    print(res)
