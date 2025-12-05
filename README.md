# 🚗 Autonomous Driving with Deep Reinforcement Learning

This project focuses on training an autonomous vehicle to navigate a simulated track by following lanes using **Deep Reinforcement Learning (DRL)**.  
The agent learns continuous control over steering and throttle using **visual observations** from the DonkeyCar simulator, trained with **PPO (Stable-Baselines3)**.

As part of the preparation for this project, I implemented several foundational RL algorithms on environments like **FrozenLake** and **MiniGrid** to build a strong understanding of value-based methods, policy learning, Monte Carlo techniques, Temporal-Difference learning, and exploration strategies before moving to vision-based continuous control .They are displayed in the Classical RL repository.

---

## 🔧 Environment: gym-donkeycar

**Actions (Continuous):**
- **Steering:** −5 to +5  
- **Throttle:** 0 to 1  

**Observations:**  
- Front-facing camera images (raw visual input)

**Rewards:**  
- Staying centered within the lane  
- Maintaining stable and appropriate speed  
- Penalties for drifting, going off-track, or crashing  

**Episode Termination:**  
- Vehicle leaves the track  
- Collision or instability  

---

## 🧠 What I Learned

- Applied DRL to a real-time, continuous-control driving task  
- Worked with visual observations and CNN-based feature extraction (through DonkeyCar + PPO architecture)  
- Tuned PPO agents using reward shaping, exploration strategies, and environment feedback  
- Understood stability considerations in RL training with high-dimensional inputs  
- Developed an autonomous control policy capable of handling dynamic driving conditions  

---

## 🏁 Project Goal

Build a self-driving RL agent that can:
- Stay centered in the lane  
- Maintain optimal speed  
- Navigate a full lap reliably without collisions  

---

