import time
from ttk_env import Env, input_to_state_map
from agent import Agent
import numpy as np

if __name__ == "__main__":
    env = Env()
    agent = Agent(symbol=1, alpha=0.1, epsilon=0.1)

    TOTAL_GAMES = 10000
    print(f"Starting training for {TOTAL_GAMES} games...")

    results = []

    for i in range(TOTAL_GAMES):
        state = env.reset()
        agent.update_history(state, was_exploratory=False)
        done = False

        while not done:
            if env.current_player == 1:
                action, was_random = agent.choose_action(state)
            else:
                possible_moves = Env.get_possible_moves(state)
                rand_idx = np.random.randint(len(possible_moves))
                action = possible_moves[rand_idx]
                was_random = False

            next_state, reward, done = env.step(action)

            agent.update_history(next_state, was_exploratory=was_random)
            state = next_state

        agent.learn()

        if env.winner == 1:
            results.append(1.0)
        elif env.winner == 0:
            results.append(0.5)
        else:
            results.append(0.0)

        if i > 0 and i % 1000 == 0:
            agent.epsilon = max(0.01, agent.epsilon * 0.9)

        if (i + 1) % 1000 == 0:
            recent = results[-1000:]
            print(
                f"Games {i + 1}: Win Rate: {recent.count(1.0) / 10:.1f}% | Loss Rate: {recent.count(0.0) / 10:.1f}%"
            )

    print("\nTraining Complete!")

    print("\n" + "=" * 40)
    print("🎮 GAME ON: YOU vs THE AGENT")
    print("Agent is X (Player 1) - You are O (Player -1)")
    print("Use keys: A, Z, E / Q, S, D / W, X, C")
    print("=" * 40 + "\n")

    agent.epsilon = 0.0

    while True:
        choice = input("Press ENTER to play (or 'q' to quit): ")
        if choice.lower() == "q":
            break

        state = env.reset()
        done = False
        env.render()

        while not done:
            if env.current_player == 1:
                print("\nAgent (X) is thinking...")
                time.sleep(0.5)
                action, _ = agent.choose_action(state)
            else:
                print(f"\nYour Turn (O)")
                valid_move = False
                while not valid_move:
                    user_input = input("Choose move: ").strip().upper()
                    if user_input not in input_to_state_map:
                        print("Invalid Key! Use A,Z,E...")
                        continue

                    action = input_to_state_map[user_input]
                    possible = Env.get_possible_moves(state)
                    board = Env.to_array(state)
                    if board[action] != 0:
                        print("Space already taken!")
                    else:
                        valid_move = True

            next_state, reward, done = env.step(action)
            env.render()
            state = next_state

        if env.winner == 1:
            print("\nAgent WINS! (It learned well)")
        elif env.winner == -1:
            print("\nYOU WIN! (Agent needs more training)")
        else:
            print("\nIt's a DRAW!")
