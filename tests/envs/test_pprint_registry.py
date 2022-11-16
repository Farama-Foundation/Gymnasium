import gymnasium as gym

# To ignore the trailing whitespaces, will need flake to ignore this file.
# flake8: noqa


def test_pprint_custom_registry():
    """Testing a registry different from default."""
    a = {
        "CartPole-v0": gym.envs.registry["CartPole-v0"],
        "CartPole-v1": gym.envs.registry["CartPole-v1"],
    }
    out = gym.pprint_registry(a, disable_print=True)

    correct_out = """===== classic_control =====
CartPole-v0 
CartPole-v1 

"""
    assert out == correct_out


def test_pprint_registry():
    """Testing the default registry, with no changes."""
    out = gym.pprint_registry(disable_print=True)

    correct_out = """===== classic_control =====
Acrobot-v1                   
CartPole-v0                  
CartPole-v1                  
MountainCar-v0               
MountainCarContinuous-v0     
Pendulum-v1                  

===== box2d =====
BipedalWalker-v3             
BipedalWalkerHardcore-v3     
CarRacing-v2                 
LunarLander-v2               
LunarLanderContinuous-v2     

===== toy_text =====
Blackjack-v1                 
CliffWalking-v0              
FrozenLake-v1                
FrozenLake8x8-v1             
Taxi-v3                      

===== mujoco =====
Ant-v2                       Ant-v3                       Ant-v4                       
HalfCheetah-v2               HalfCheetah-v3               HalfCheetah-v4               
Hopper-v2                    Hopper-v3                    Hopper-v4                    
Humanoid-v2                  Humanoid-v3                  Humanoid-v4                  
HumanoidStandup-v2           HumanoidStandup-v4           InvertedDoublePendulum-v2    
InvertedDoublePendulum-v4    InvertedPendulum-v2          InvertedPendulum-v4          
Pusher-v2                    Pusher-v4                    Reacher-v2                   
Reacher-v4                   Swimmer-v2                   Swimmer-v3                   
Swimmer-v4                   Walker2d-v2                  Walker2d-v3                  
Walker2d-v4                  

===== external =====
GymV26Environment-v0         

===== utils_envs =====
RegisterDuringMakeEnv-v0     
test.ArgumentEnv-v0          
test.OrderlessArgumentEnv-v0 

===== test =====
test/NoHuman-v0              
test/NoHumanNoRGB-v0         
test/NoHumanOldAPI-v0        

"""
    assert out == correct_out


def test_pprint_registry_exclude_namespaces():
    """Testing the default registry, with no changes."""
    out = gym.pprint_registry(
        max_rows=20, exclude_namespaces=["classic_control"], disable_print=True
    )

    correct_out = """===== box2d =====
BipedalWalker-v3             
BipedalWalkerHardcore-v3     
CarRacing-v2                 
LunarLander-v2               
LunarLanderContinuous-v2     

===== toy_text =====
Blackjack-v1                 
CliffWalking-v0              
FrozenLake-v1                
FrozenLake8x8-v1             
Taxi-v3                      

===== mujoco =====
Ant-v2                       Ant-v3                       
Ant-v4                       HalfCheetah-v2               
HalfCheetah-v3               HalfCheetah-v4               
Hopper-v2                    Hopper-v3                    
Hopper-v4                    Humanoid-v2                  
Humanoid-v3                  Humanoid-v4                  
HumanoidStandup-v2           HumanoidStandup-v4           
InvertedDoublePendulum-v2    InvertedDoublePendulum-v4    
InvertedPendulum-v2          InvertedPendulum-v4          
Pusher-v2                    Pusher-v4                    
Reacher-v2                   Reacher-v4                   
Swimmer-v2                   Swimmer-v3                   
Swimmer-v4                   Walker2d-v2                  
Walker2d-v3                  Walker2d-v4                  

===== external =====
GymV26Environment-v0         

===== utils_envs =====
RegisterDuringMakeEnv-v0     
test.ArgumentEnv-v0          
test.OrderlessArgumentEnv-v0 

===== test =====
test/NoHuman-v0              
test/NoHumanNoRGB-v0         
test/NoHumanOldAPI-v0        

"""
    assert out == correct_out


def test_pprint_registry_no_entry_point():
    """Test registry if there is environment with no entry point."""

    gym.register("NoNamespaceEnv", "no-entry-point")
    out = gym.pprint_registry(disable_print=True)

    correct_out = """===== classic_control =====
Acrobot-v1                   
CartPole-v0                  
CartPole-v1                  
MountainCar-v0               
MountainCarContinuous-v0     
Pendulum-v1                  

===== box2d =====
BipedalWalker-v3             
BipedalWalkerHardcore-v3     
CarRacing-v2                 
LunarLander-v2               
LunarLanderContinuous-v2     

===== toy_text =====
Blackjack-v1                 
CliffWalking-v0              
FrozenLake-v1                
FrozenLake8x8-v1             
Taxi-v3                      

===== mujoco =====
Ant-v2                       Ant-v3                       Ant-v4                       
HalfCheetah-v2               HalfCheetah-v3               HalfCheetah-v4               
Hopper-v2                    Hopper-v3                    Hopper-v4                    
Humanoid-v2                  Humanoid-v3                  Humanoid-v4                  
HumanoidStandup-v2           HumanoidStandup-v4           InvertedDoublePendulum-v2    
InvertedDoublePendulum-v4    InvertedPendulum-v2          InvertedPendulum-v4          
Pusher-v2                    Pusher-v4                    Reacher-v2                   
Reacher-v4                   Swimmer-v2                   Swimmer-v3                   
Swimmer-v4                   Walker2d-v2                  Walker2d-v3                  
Walker2d-v4                  

===== external =====
GymV26Environment-v0         

===== utils_envs =====
RegisterDuringMakeEnv-v0     
test.ArgumentEnv-v0          
test.OrderlessArgumentEnv-v0 

===== test =====
test/NoHuman-v0              
test/NoHumanNoRGB-v0         
test/NoHumanOldAPI-v0        

===== NoNamespaceEnv =====
NoNamespaceEnv               

"""
    assert out == correct_out

    del gym.envs.registry["NoNamespaceEnv"]
