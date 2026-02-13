import gymnasium as gym
import gym_platform
import numpy as np
import sys

def main():
    print("Initializing Platform-v0 with render_mode='human'")
    # Create the environment
    try:
        env = gym.make('Platform-v0', render_mode='human')
    except Exception as e:
        print(f"Failed to create environment: {e}")
        return

    obs, info = env.reset()
    print("Environment reset.")
    print("\n--- Platform-v0 Manual Control ---")
    print("Available Actions:")
    print("0: RUN  (Parameter range: 0.0 - 30.0)")
    print("1: HOP  (Parameter range: 0.0 - 720.0)")
    print("2: LEAP (Parameter range: 0.0 - 430.0)")
    print("q: Quit")
    print("-" * 50)
    print("Enter combined command: <action_id> <parameter>")
    print("Example: '0 20.0' to Run with power 20.0")

    # Min/Max for clamping/guidance
    # Based on platform_env.py constants
    min_params = [0.0, 0.0, 0.0]
    # RUN, HOP, LEAP maxes
    max_params = [30.0, 720.0, 430.0]
    action_names = ["RUN", "HOP", "LEAP"]

    while True:
        try:
            user_input = input("\nAction (0-2) and Param, or 'q': ").strip()
        except KeyboardInterrupt:
            print("\nExiting...")
            break
            
        if user_input.lower() == 'q':
            break
            
        parts = user_input.split()
        if len(parts) != 2:
            print("Error: Input must be two numbers separated by space.")
            continue
            
        try:
            act_idx = int(parts[0])
            param_val = float(parts[1])
        except ValueError:
            print("Error: Could not parse numbers.")
            continue
            
        if act_idx not in [0, 1, 2]:
            print("Error: Action must be 0, 1, or 2.")
            continue
            
        # Optional: Warn if out of bounds (Env clips it anyway)
        if param_val < min_params[act_idx] or param_val > max_params[act_idx]:
            print(f"Note: Parameter {param_val} will be clipped to [{min_params[act_idx]}, {max_params[act_idx]}] by the environment.")

        print(f"Step: {action_names[act_idx]} ({param_val})")
        
        # Prepare action for gymnasium environment
        # Action space is Tuple(Discrete(3), Tuple(Box(1), Box(1), Box(1)))
        # We need to construct the full structure.
        
        # Create parameter arrays for all actions (defaults to 0 or min)
        # We only care about the one at act_idx, but structure must be complete.
        action_params = [np.array([min_params[i]], dtype=np.float32) for i in range(3)]
        
        # Set the specific parameter for the chosen action
        action_params[act_idx][0] = param_val
        
        # Combine into the tuple expected by the step function
        # action = (int_index, (p1_array, p2_array, p3_array))
        action = (act_idx, tuple(action_params))
        
        # Execute Step
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render the result (animation of the step)
            env.render()
            
            print(f"Result -> Reward: {reward:.4f}, Terminated: {terminated}, Truncated: {truncated}")
            
            if terminated or truncated:
                print(">> Episode Ended. Resetting environment.")
                obs, info = env.reset()
                
        except Exception as e:
            print(f"Error during step/render: {e}")
            import traceback
            traceback.print_exc()

    print("Closing environment.")
    env.close()

if __name__ == "__main__":
    main()
