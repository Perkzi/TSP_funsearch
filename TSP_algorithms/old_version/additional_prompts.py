# 单步的prompt
base_prompt = (
            "Complete a different and more complex Python function. "
            "Be creative and you can insert multiple if-else and for-loop in the code logic. "
            "Only output the Python code, no descriptions."
        )
'''
这种提示引导模型生成一个结构化、分阶段的解决方案，适合希望看到详细过程、并便于调试和维护的代码。
逐步拆解有助于对算法进行深入检查，但可能会使代码较长，适合教学或实验阶段
'''
step_by_step_prompt = (
    "Complete a different and more complex Python function that solves the TSP problem "
    "by breaking down the algorithm into clear, sequential steps. "
    "Separate the process into initialization, iterative improvement, and termination phases. "
    "You may use loops, recursion, and conditional statements to express each step. "
    "Only output the Python code, no descriptions."
)
'''
这种提示鼓励模型采用多条不同的算法路径进行比较，更适用于需要探索不同算法策略（例如分治法与启发式搜索结合）的场景。
缺点可能在于实现起来逻辑较为复杂，需要在实际开发中仔细调试各个分支的协同工作。'''
multi_path_analysis_prompt = (
    "Complete a different and more complex Python function for the TSP problem. "
    "Internally consider multiple solution paths using different heuristics or branching strategies, "
    "but select and output only one final integrated Python code version as the result. "
    "Do not output multiple pieces of code or any chain-of-thought explanations; "
    "only output the final Python code."
)
'''
这种方式融合了常规启发式方法与元启发式优化，适用于希望在局部与全局搜索之间取得平衡的应用情况。
虽然这种混合方法能产生较强的鲁棒性，但复杂度较高，需要仔细调优参数和流程设计
'''
heuristic_meta_prompt = (
    "Complete a different and more complex Python function for solving the TSP problem by integrating "
    "heuristic methods with metaheuristic optimization techniques. For example, combine greedy selection, "
    "simulated annealing, or genetic algorithm-inspired strategies within your logic. "
    "Use loops, conditional statements, or recursion as necessary. "
    "Only output the Python code, no descriptions."
)
'''
将算法分为若干模块或阶段，有助于提高代码的可读性与可维护性，尤其是在复杂问题中分块处理各阶段逻辑。
这种方法利于后续增删改动，但对模块边界和接口的设计要求较高。
'''
multi_phase_prompt = (
    "Complete a different and more complex Python function for the TSP problem that implements a multi-phase algorithm. "
    "Divide the solution into distinct phases such as initial tour construction, local search optimization, and global refinement cycles. "
    "Utilize creative control flows (loops, conditionals, recursion) to clearly separate and manage each phase. "
    "Only output the Python code, no descriptions."
)

self_analysis_prompt = (
    "Please first provide a detailed explanation of the implementation strategy for the following heuristic TSP code,"
    " highlighting its limitations and potential failure cases (including a specific example)."
    " Then, based on your analysis, propose an improved approach and rewrite the code accordingly."
    " In your final output, only include the final, consolidated Python code without any additional explanations or text."
)