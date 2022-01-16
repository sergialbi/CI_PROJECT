# CI project

## Dependencies

First of all, you will need to install the following libraries to execute our code:

* scikit-learn
* tensorflow
* keras
* tensorboard
* tensorflow-probabilities
* box2d
* PyTorch
* Gym

## How to execute our code

* Step 1: Open the folder CI_PROJECT and find main.py

* Step 2: Write on your terminal py main.py

* Step 3: The following messages will appear on your terminal. Select the option that you want to execute. 

  ```
  0: create folders
  1: run Deep Q Learning algorithm
  2: run PPO algorithm
  3: run Genetic algorithm
  ```

  ***Note**: If it is your first time you will need to execute first the option 0* 

* Step 4: Once the option is selected, select the environment that you prefer to use to execute the selected algorithm.

  ```
  Select game: w (Bipedal Walker) - c (CartPole)
  ```

* Step 4.2 (only with PPO): Select if you want to either train or test the algorithm with the selected environment.

  ```
  Select mode: t (training) - p (test)
  ```

  To test you will need to uncomment the best found parameters, which can be found in constants/contants_ppo.py

* Step 5: Wait for the execution to finish

Results can be found on results folder.

Also, you can use tensorbord to visualize the training curves for each environment. To do it, you need to open a 
terminal in the results/ppo folder, and type
```
tensorbord --logdir CartPole-v1 (or BipedalWalker-v3) 
```