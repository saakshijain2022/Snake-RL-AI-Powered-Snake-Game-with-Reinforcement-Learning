from graphviz import Digraph

def create_uml_diagram():
    dot = Digraph()

    # Define nodes for each class
    dot.node('SG', 'SnakeGameAI')
    dot.node('A', 'Agent')
    dot.node('LQ', 'Linear_QNet')
    dot.node('QT', 'QTrainer')

    # Define attributes for each class
    dot.node('SGA', 'Attributes:\n- w\n- h\n- display\n- clock\n- direction\n- head\n- snake\n- score\n- food\n- frame_iteration', shape='box')
    dot.node('AA', 'Attributes:\n- n_games\n- epsilon\n- gamma\n- memory\n- model\n- trainer', shape='box')
    dot.node('LQA', 'Attributes:\n- linear1\n- linear2', shape='box')
    dot.node('QTA', 'Attributes:\n- lr\n- gamma\n- model\n- optimizer\n- criterion', shape='box')

    # Define methods for each class
    dot.node('SGM', 'Methods:\n- reset()\n- _place_food()\n- play_step(action)\n- is_collision(pt)\n- _update_ui()\n- _move(action)', shape='box')
    dot.node('AM', 'Methods:\n- get_state(game)\n- remember(state, action, reward, next_state, done)\n- train_long_memory()\n- train_short_memory()\n- get_action(state)', shape='box')
    dot.node('LQM', 'Methods:\n- forward(x)\n- save(file_name)', shape='box')
    dot.node('QTM', 'Methods:\n- train_step(state, action, reward, next_state, done)', shape='box')

    # Create edges between classes
    dot.edge('A', 'SG', label='interacts with')
    dot.edge('A', 'LQ', label='uses')
    dot.edge('QT', 'LQ', label='trains')
    
    # Add class nodes with their attributes and methods
    dot.node('SG', 'SnakeGameAI\n' + 'Attributes:\n- w\n- h\n- display\n- clock\n- direction\n- head\n- snake\n- score\n- food\n- frame_iteration\n' + 'Methods:\n- reset()\n- _place_food()\n- play_step(action)\n- is_collision(pt)\n- _update_ui()\n- _move(action)', shape='rect')
    dot.node('A', 'Agent\n' + 'Attributes:\n- n_games\n- epsilon\n- gamma\n- memory\n- model\n- trainer\n' + 'Methods:\n- get_state(game)\n- remember(state, action, reward, next_state, done)\n- train_long_memory()\n- train_short_memory()\n- get_action(state)', shape='rect')
    dot.node('LQ', 'Linear_QNet\n' + 'Attributes:\n- linear1\n- linear2\n' + 'Methods:\n- forward(x)\n- save(file_name)', shape='rect')
    dot.node('QT', 'QTrainer\n' + 'Attributes:\n- lr\n- gamma\n- model\n- optimizer\n- criterion\n' + 'Methods:\n- train_step(state, action, reward, next_state, done)', shape='rect')

    # Render the UML diagram
    dot.render('snake_ai_uml', format='png', cleanup=True)

if __name__ == '__main__':
    create_uml_diagram()
