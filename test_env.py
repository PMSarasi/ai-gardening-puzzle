# test_env.py
import sys
sys.path.append('.')

print("Testing Garden Environment with Gymnasium...")

try:
    from environment.garden_env import GardenEnv
    
    # Create environment
    env = GardenEnv(grid_size=3)
    print("✓ Environment created")
    
    # Test reset
    obs, info = env.reset()
    print(f"✓ Reset successful - observation shape: {obs.shape}")
    
    # Test step
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(f"✓ Step successful - reward: {reward:.2f}")
    
    # Test a few random steps
    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        print(f"  Step {step+1}: Action={action}, Reward={reward:.2f}")
    
    print(f"\n✓✓✓ Test complete! Total reward: {total_reward:.2f}")
    print("Your garden environment is working perfectly with Gymnasium!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()