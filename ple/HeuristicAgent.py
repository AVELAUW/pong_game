# Functions for Q-Learning training!
class HeuristicAgent:
  """
      This is our agent. It decideds our actions!
  """
  
  def __init__(self, actions, divisor):
    self.decisions = {}
    self.divisor = divisor
    self.actions = list(actions)[:len(actions)-1] # eliminate 'none' action

  def pick_action(self, gameState):
    simple_gameState = (np.round(gameState['player_y']/self.divisor), np.round(gameState['ball_y']/self.divisor), np.round(gameState['ball_x']/self.divisor))
    if simple_gameState in self.decisions:
      return self.decisions[simple_gameState]
    else:
      return random.choice(self.actions)

  def q_learning(self, learning_rate=1, discount_factor=0.99, epsilon=0.05, episodes=1000, strategy="smart", screen_w=512, screen_h=384, cpu_speed_ratio=0.5, players_speed_ratio = 0.5, ball_speed_ratio=0.75,  max_score=3):
    # Q-learning variables
    Q = {}
    agent_combinations = {}
    alpha, discount = learning_rate, discount_factor
    scores = []
    # Game variables
    game = Pong(screen_w,screen_h,cpu_speed_ratio,players_speed_ratio,MAX_SCORE=max_score) # 384,288
    p = PLE(game, fps=30, display_screen=False)
    p.init()
    if (not (strategy=="random")):
      for episode in range(episodes):
        p.reset_game()
        while not p.game_over():
          agent_state = (np.round(p.getGameState()['player_y']/self.divisor), np.round(p.getGameState()['ball_y']/self.divisor), np.round(p.getGameState()['ball_x']/self.divisor))
          agent_action = get_action(strategy, Q, agent_combinations, agent_state, self.actions, epsilon)
          agent_combinations[agent_state, agent_action] = True
          p.act(agent_action)
          reward = game.getReward()
          next_state = (np.round(p.getGameState()['player_y']/self.divisor), np.round(p.getGameState()['ball_y']/self.divisor), np.round(p.getGameState()['ball_x']/self.divisor))
          next_action = best_action(strategy, Q, next_state, self.actions)
          
          if (agent_state, agent_action) not in Q:
            Q[agent_state, agent_action] = 0
          if (next_state, next_action) not in Q:
            Q[next_state, next_action] = 0

          Q[agent_state, agent_action] = (1-alpha)*Q[agent_state,agent_action] + alpha*(reward+discount*Q[next_state,next_action])
        
      for states in Q:
        if states[0] not in self.decisions:
          self.decisions[states[0]] = states[1]
        else:
          if (Q[states[0],self.decisions[states[0]]] < Q[states]):
            self.decisions[states[0]] = states[1]

def get_action(player_strategy, Q, seen_combinations, state, actions, epsilon):
  if player_strategy == "dumb":
    return 0
  else:
    return epsilon_greedy(player_strategy, Q, seen_combinations, state, actions, epsilon)  

def epsilon_greedy(player_strategy, Q, seen_combinations, state, possible_actions, epsilon):
  not_tried_yet = []
  for action in possible_actions:
      if (state, action) not in seen_combinations:
          not_tried_yet.append(action)
  if not_tried_yet != []:
      return random.choice(not_tried_yet)
  if random.random() < epsilon:
      return random.choice(possible_actions)
  else:
      return best_action(player_strategy, Q, state, possible_actions)
  
def best_action(player_strategy, Q, state, possible_actions):
  best_action = None
  best_action_reward = -float('inf')
  if player_strategy == "dumb":
    return 0
  for action in possible_actions:
      if (state, action) not in Q:
          Q[state, action] = 0
      if Q[state, action] > best_action_reward:
          best_action_reward = Q[state, action]
          best_action = action
  return best_action
