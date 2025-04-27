#

base_prompt = (
            "Complete a different and more complex Python function. "
            "Be creative and you can insert multiple if-else and for-loop in the code logic. "
            "Only output the Python code without any additional explanations. "
            "Do not alter the input/output logic of the tsp_priority function (ensure it only returns a single float). "
            "Do not modify the parameter description in the tsp_priority function."
        )


tsp_specific_prompt = (
            "Complete a different and more complex Python function to solve TSP problem. "
            "You have to travel like a circle to visit all cities and finally go back to the start city."
            "For example, if you always choose the nearest, you will have to pay a lot for going back to the start city"
            "Be creative and you can insert multiple if-else and for-loop in the code logic. "
            "Only output the Python code without any additional explanations. "
            "Do not alter the input/output logic of the tsp_priority function (ensure it only returns a single float). "
            "Do not modify the parameter description in the tsp_priority function."
        )