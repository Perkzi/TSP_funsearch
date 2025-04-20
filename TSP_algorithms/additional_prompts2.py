# 把prompt分为两步

'''
这种提示引导模型生成一个结构化、分阶段的解决方案，适合希望看到详细过程、并便于调试和维护的代码。
逐步拆解有助于对算法进行深入检查，但可能会使代码较长，适合教学或实验阶段
'''
step_by_step_prompts = [
    (
        "Analyze the current heuristic approach for solving the TSP problem and outline a structured, step-by-step optimization logic. "
        "Break down the algorithm into clear phases such as initialization, iterative improvement, and termination, and discuss potential pitfalls and limitations. "
        "Propose ways to improve the algorithm using control structures like loops, recursion, and conditionals. "
        "Only output the analysis text, no code."
    ),
    (
        "Complete a different and more complex Python function that solves the TSP problem. "
        "Incorporate the following optimization logic into your solution: <insert optimization analysis here> "
        "and generate a final, integrated version of the code that reflects the suggested improvements. "
        "Only output the Python code, no other descriptions."
    )
]
'''
这种提示鼓励模型采用多条不同的算法路径进行比较，更适用于需要探索不同算法策略（例如分治法与启发式搜索结合）的场景。
缺点可能在于实现起来逻辑较为复杂，需要在实际开发中仔细调试各个分支的协同工作。'''
multi_path_analysis_prompts = [
    # 第一步：要求模型分析多路径策略、讨论各自优缺点，提供可能失败的例子
    (
        "Analyze the TSP problem's current heuristic approach by considering multiple solution paths that use different heuristics "
        "or branching strategies (e.g., divide-and-conquer vs. heuristic search). Discuss the benefits and potential pitfalls of each approach, "
        "and provide an example where the algorithm might fail. Only output the analysis text, no code."
    ),
    # 第二步：要求模型根据第一步的分析整合并生成最后的 Python 代码
    (
        "Complete a different and more complex Python function for the TSP problem. "
        "Incorporate the following optimization logic into your solution: <insert multi-path analysis here>. "
        "Select and output only one final integrated version of the Python code, reflecting the analysis above. "
        "Only output the final Python code, no explanations."
    )
]
'''
这种方式融合了常规启发式方法与元启发式优化，适用于希望在局部与全局搜索之间取得平衡的应用情况。
虽然这种混合方法能产生较强的鲁棒性，但复杂度较高，需要仔细调优参数和流程设计
'''
heuristic_meta_prompts = [
    # 第一步：让模型分析融合启发式与元启发式策略的优势、局限以及调优需求
    (
        "Analyze the current approach that integrates heuristic methods with metaheuristic optimization techniques for solving the TSP problem. "
        "Discuss the balance between local and global search (e.g., using greedy selection, simulated annealing, or genetic algorithms), "
        "highlight potential limitations and areas requiring parameter or process adjustments. "
        "Only output the analysis text, no code."
    ),
    # 第二步：要求模型根据分析内容生成改进后的代码
    (
        "Complete a different and more complex Python function for solving the TSP problem by integrating heuristic methods with metaheuristic techniques. "
        "Incorporate the following analysis into your solution: <insert heuristic-meta analysis here>. "
        "Generate a final, integrated Python code version that addresses the discussed limitations. "
        "Only output the Python code, no explanations."
    )
]

'''
将算法分为若干模块或阶段，有助于提高代码的可读性与可维护性，尤其是在复杂问题中分块处理各阶段逻辑。
这种方法利于后续增删改动，但对模块边界和接口的设计要求较高。
'''
multi_phase_prompts = [
    # 第一步：要求模型分析将算法分为多个阶段的优势及注意事项
    (
        "Analyze the benefits and potential pitfalls of adopting a multi-phase algorithm approach for the TSP problem. "
        "Break down the solution into distinct phases (e.g., initial tour construction, local search optimization, and global refinement). "
        "Discuss how this modular design can aid code readability and maintainability, and mention any challenges regarding module interfaces. "
        "Only output the analysis text, no code."
    ),
    # 第二步：要求模型生成整合各阶段逻辑的最终代码
    (
        "Complete a different and more complex Python function for solving the TSP problem that implements a multi-phase algorithm. "
        "Incorporate the following analysis into your solution: <insert multi-phase analysis here>. "
        "Generate a final, integrated version of the Python code that clearly separates and manages each phase. "
        "Only output the Python code, no explanations."
    )
]

self_analysis_prompts = [
    # 第一步：要求模型详细解释当前启发式 TSP 代码的策略、局限性及失败实例
    (
        "Please analyze the given heuristic TSP code by outlining its implementation strategy, limitations, and potential failure cases (including a specific example). "
        "Discuss how the algorithm might be improved. Only output your analysis text, no code."
    ),
    # 第二步：要求模型基于上述分析生成改进后的代码
    (
        "Complete a different and more complex Python function for the TSP problem. "
        "Based on the following analysis: <insert self-analysis here>, "
        "rewrite and improve the code to address the identified issues. "
        "Only output the final, consolidated Python code, no additional explanations."
    )
]
