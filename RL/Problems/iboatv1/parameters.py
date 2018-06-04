
ENV = "Pendulum-v0"

LOAD = False
DISPLAY = True


DISCOUNT = 0.99

FRAME_SKIP = 0


ACTOR_LEARNING_RATE = 7e-4
CRITIC_LEARNING_RATE = 5e-5

# Memory size
BUFFER_SIZE = 100000
BATCH_SIZE = 32

# Number of episodes of game environment to train with
TRAINING_STEPS = 4000

# Maximal number of steps during one episode
MAX_EPISODE_STEPS = 30 #200*4
TRAINING_FREQ = 1

# Rate to update target network toward primary network
UPDATE_TARGET_RATE = 0.02

# scale of the exploration noise process (1.0 is the range of each action
# dimension)
NOISE_SCALE_INIT = 0.1

# decay rate (per episode) of the scale of the exploration noise process
NOISE_DECAY = 0.99

# parameters for the exploration noise process:
# dXt = theta*(mu-Xt)*dt + sigma*dWt
EXPLO_MU = 0.0
EXPLO_THETA = 0.15
EXPLO_SIGMA = 0.2

# Display Frequencies
DISP_EP_REWARD_FREQ = 10
PLOT_FREQ = 100
RENDER_FREQ = 100
