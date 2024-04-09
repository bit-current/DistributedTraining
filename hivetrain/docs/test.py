from graphviz import Digraph

dot = Digraph(comment='Architectural Diagram')

dot.node('A', 'TrainingLoop')
dot.node('B', 'LocalTrainingLoop')
dot.node('C', 'torch')
dot.node('D', 'transformers')
dot.node('E', 'huggingface_hub')
dot.node('F', 'time')

dot.edges(['AB', 'AC', 'AD', 'AE', 'AF'])
dot.edge('B', 'C', constraint='false')

dot.render('training_loop_architecture', view=True)