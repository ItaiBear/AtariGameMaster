# AtariGameMaster

Itai Bear, 325839710, itai.bear1@gmail.com   
Noam Manaker Morag, 214726002, noam.school.only@gmail.com

## Progress Report
### December
We took off our learning period by going over the RL Course by David Silver: https://www.davidsilver.uk/teaching/   
We also started the Deep Reinforcement Learning course by Huggingface, which had practical programming assignments: https://huggingface.co/deep-rl-course/unit0/introduction?fw=pt

### January
We finished the the course by David Silver.    
Due to the exam period we werent able to finish the Huggingface course, but we are on our way to do so.

### February
We finished the Hugginface course, began search for what game to solve. We are searching for a game that is sufficiently complex to be interesting and also has a gym environment.

### March
Found an existing tetris emulator with a prepared gym-environment. Began training of a Deep Q Learning model to play the game.    
Tried using some simple reward functions:
- In-Game score
- Number of lines cleared
- Penalization for board height
But none of them yieled signifciant results.  

### April

Began using Optuna, a library for hyper-parameter opitimization. Optuna tries many different combinations of hyper-parameters and selects the best one.  

Tried more advanced reward functions:
- Penalization for empty cells
- "Bumpiness"
- Different combinations of the different rewards 
Unfortanely none of these worked either.  

We began searching for different ideas to improve the model.

### May
