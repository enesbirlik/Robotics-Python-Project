import matplotlib.pyplot as plt

class RobotVisualizer:
    @staticmethod
    def plot_solution(robot, q, title="Robot Configuration"):
        plt.figure()
        robot.robot.plot(q)
        plt.title(title)
        plt.show()
    
    @staticmethod
    def compare_solutions(robot, solutions_dict):
        num_solutions = len(solutions_dict)
        plt.figure(figsize=(5*num_solutions, 5))
        
        for i, (name, q) in enumerate(solutions_dict.items(), 1):
            plt.subplot(1, num_solutions, i)
            robot.robot.plot(q)
            plt.title(f'{name} Solution')
        
        plt.show()