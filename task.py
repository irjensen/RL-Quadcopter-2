import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=[0., 0., 50.], 
                 init_angle_velocities=None, runtime=5., target_height=100.):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 1

        self.state_size = self.action_repeat * 9
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_height = target_height
        
    def get_reward(self):
        """Uses current pose of sim to return reward."""
        #reward = self.sim.v[2]**2
        #reward = 0.0
        reward = 10.0/(1+abs(self.sim.pose[2]-self.target_height))**2
        #reward += 100./(1+abs(np.linalg.norm(self.sim.pose[:2])-self.target_circle_radius))
        #reward -= 10.0*np.linalg.norm(self.sim.pose[6:9])**2
#         reward += 1*np.linalg.norm(self.sim.pose[3:6])**2 if self.is_in_circle() else -0.5
#         reward = 1 - np.tanh(abs(150 - self.sim.pose[2])/2 + abs(self.sim.v[2]/2))
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
#             if done:
#                 pass
                #reward = -500
            reward += self.get_reward() 
            pose_all.append(np.concatenate((self.sim.pose, self.sim.v),axis=None))
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([np.concatenate((self.sim.pose, self.sim.v),axis=None)] * self.action_repeat) 
        return state