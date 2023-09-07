import requests
import numpy as np

# Create class object for a single linear ucb disjoint arm
# update BatchLinearUCB by advice from chatGPT


# If I were using linear ucb algorith, it should calculate inverse of matrix with dimension of (number of context fetures, number of context fetures). It takes long time. Is it inevitable?
# ChatGPT
# The Linear UCB (Upper Confidence Bound) algorithm does indeed require the inversion of a matrix with dimensions corresponding to the number of context features. Specifically, if you're dealing with 
# d context features, then you'll need to frequently invert a 
# d×d matrix as part of the algorithm. 
# 
# In your case, for a # 32# × # 32 # × # 1 # 32×32×1 image, this corresponds to a  # 1024# ×# 1024  1024×1024 matrix.

# Matrix inversion is an 
# �
# (
# �
# 3
# )
# O(d 
# 3
#  ) operation, which can become computationally intensive as 
# �
# d grows. Here are some strategies and considerations to address this challenge:

# Matrix Inversion Lemma (Sherman-Morrison formula): When updating the matrix with a new observation, instead of recalculating the entire matrix inversion, you can use the Matrix Inversion Lemma to update the inverse based on the previous inverse. This can be significantly faster than direct inversion, especially when updates are frequent.

# Regularization: Adding a regularization term (e.g., ridge regression) will ensure that the matrix remains invertible even if the features are collinear. It also stabilizes the matrix inversion.

# Feature Reduction: While your images are already relatively small, if the computational cost is still too high, consider using dimensionality reduction techniques like PCA or an autoencoder. This would reduce the number of features (and hence the matrix size) while retaining most of the image's essential information.

# Alternative Algorithms: There are other contextual bandit algorithms or variations that might be less computationally intensive than the Linear UCB. For example, consider algorithms that don't rely on matrix inversion or look into approximate methods.

# Online Learning Libraries: Several libraries are optimized for online learning and might have efficient implementations of Linear UCB. Libraries like Vowpal Wabbit or scikit-learn's online learning modules might be worth exploring.

# Sparse Matrix Representation: If your context (image in this case) leads to a sparse matrix, leveraging sparse matrix representations and operations can significantly speed up computations.

# Batch Updates: Instead of updating the model with every single data point, you can accumulate data and update the model in batches. This can be more computationally efficient, though it may slow down the adaptation speed of the bandit.

# In conclusion, while the matrix inversion in Linear UCB can be computationally intensive, especially for large contexts, there are techniques and strategies to mitigate these computational costs. Choosing the right approach depends on the specific application, the available computational resources, and the desired trade-off between accuracy and computational efficiency.


# import numpy as np

# class BatchLinearUCB:
#     def __init__(self, d, alpha, batch_size):
#         self.d = d  # dimension of context features
#         self.alpha = alpha  # exploration parameter
#         self.batch_size = batch_size
        
#         self.A = np.eye(d)  # Identity matrix for initialization
#         self.b = np.zeros((d, 1))
        
#         # Buffers for batch update
#         self.context_buffer = []
#         self.reward_buffer = []
        
#     def predict(self, context):
#         # Predict reward using current model
#         theta = np.linalg.inv(self.A).dot(self.b)
#         p = context.dot(theta) + self.alpha * np.sqrt(np.diagonal(context.dot(np.linalg.inv(self.A)).dot(context.T)))
#         return p

#     def add_to_buffer(self, context, reward):
#         self.context_buffer.append(context)
#         self.reward_buffer.append(reward)
        
#         # Check if buffer is full and needs updating
#         if len(self.context_buffer) == self.batch_size:
#             self.update_model()
            
#     def update_model(self):
#         # Batch update the A matrix and b vector
#         for i in range(self.batch_size):
#             context = self.context_buffer[i]
#             reward = self.reward_buffer[i]
#             self.A += np.outer(context, context)
#             self.b += context * reward
        
#         # Clear the buffer
#         self.context_buffer = []
#         self.reward_buffer = []

# # Sample usage
# d = 1024  # for 32x32 image flattened
# alpha = 0.1
# batch_size = 50

# model = BatchLinearUCB(d, alpha, batch_size)

# # Sample data
# context = np.random.randn(d)
# reward = np.random.randn(1)

# for _ in range(100):  # Simulate 100 data points
#     model.add_to_buffer(context, reward)
#     pred = model.predict(context)
# This code defines a BatchLinearUCB class that stores a buffer of context and reward data. When the buffer is full (reaches the defined batch_size), the model updates its A matrix and b vector using all the buffered data. After the update, the buffer is cleared to prepare for new data.

# Note: The matrix inversion (np.linalg.inv(self.A)) occurs during prediction. If this is a bottleneck, it can be optimized using techniques like the Sherman-Morrison formula, as mentioned earlier.


class linucb_disjoint_arm():
    # TODO: 여기 batch_size는 실제 메인 모델인 fixmatch의 배치사이즈와 다를수 있으나, 현재는 동일하게 사용하기로.
    def __init__(self, arm_index, d, alpha, batch_size):
        # Track arm index
        self.arm_index = arm_index
        # Keep track of alpha
        self.alpha = alpha
        # A: (d x d) matrix = D_a.T * D_a + I_d. 
        # The inverse of A is used in ridge regression 
        self.A = np.identity(d)
        # b: (d x 1) corresponding response vector. 
        # Equals to D_a.T * c_a in ridge regression formulation
        self.b = np.zeros([d,1])
        # Find A inverse for ridge regression
        self.A_inv = np.linalg.inv(self.A)
        self.batch_size = batch_size
        self.context_buffer = []
        self.reward_buffer = []
        
    def calc_UCB(self, x_array):
        # Perform ridge regression to obtain estimate of covariate coefficients theta
        # theta is (d x 1) dimension vector
        self.theta = np.dot(self.A_inv, self.b)
        # Reshape covariates input into (d x 1) shape vector
        x = x_array.reshape([-1,1])
        # Find ucb based on p formulation (mean + std_dev) 
        # p is (1 x 1) dimension vector
        p = np.dot(self.theta.T,x) +  self.alpha * np.sqrt(np.dot(x.T, np.dot(self.A_inv,x)))
        return p

    def add_to_buffer(self, context, reward):
        self.context_buffer.extend(context)
        self.reward_buffer.extend(reward)
        #TODO: 여긴 배치 사이즈라고 하긴 그렇군.. 최소학습되는 threshold 
        if len(self.context_buffer) >= self.batch_size:
            self.reward_update()

    def reward_update(self):
        # Batch update the A matrix and b vector
        for i in range(len(self.context_buffer)):
            context = self.context_buffer[i]
            reward = self.reward_buffer[i]
            self.A += np.outer(context, context)
            self.b += context.reshape(-1, 1) * reward
        
        # TODO: 여기 예측 시점에서하는 것을 batch update시에만 하도록 -> 성능 증가!!
        self.A_inv = np.linalg.inv(self.A)

        # Clear the buffer
        self.context_buffer = []
        self.reward_buffer = []



    # def reward_update(self, reward, x_array):
    #     # Batch update
    #     for i in range(self.batch_size):
    #         context = self.context_buffer[i]
    #         reward = self.reward_buffer[i]

    #     # Reshape covariates input into (d x 1) shape vector
    #     x = x_array.reshape([-1,1])
    #     # Update A which is (d * d) matrix.
    #     self.A += np.dot(x, x.T)
    #     # Update b which is (d x 1) vector
    #     # reward is scalar
    #     self.b += reward * x
    #     # update A_inv also
    #     self.A_inv = np.linalg.inv(self.A)


class linucb_policy():
    def __init__(self, K_arms, d, alpha, batch_size):
        self.K_arms = K_arms
        self.batch_size = batch_size
        # TODO: disjoint가 좋은지 hybrid arm이 좋은지 둘다 실험을 해볼 것
        self.linucb_arms = [linucb_disjoint_arm(arm_index = i, d = d, alpha = alpha, batch_size= batch_size) for i in range(K_arms)]
        
    def select_arm(self, x_array):
        # Initiate ucb to be 0
        highest_ucb = -1
        # Track index of arms to be selected on if they have the max UCB.
        candidate_arms = []
        
        for arm_index in range(self.K_arms):
            # Calculate ucb based on each arm using current covariates at time t
            arm_ucb = self.linucb_arms[arm_index].calc_UCB(x_array)
            # If current arm is highest than current highest_ucb
            if arm_ucb > highest_ucb:
                # Set new max ucb
                highest_ucb = arm_ucb
                # Reset candidate_arms list with new entry based on current arm
                candidate_arms = [arm_index]

            # If there is a tie, append to candidate_arms
            if arm_ucb == highest_ucb:
                candidate_arms.append(arm_index)
        
        # Choose based on candidate_arms randomly (tie breaker)
        chosen_arm = np.random.choice(candidate_arms)
        
        return chosen_arm        


def ctr_simulator(K_arms, d, alpha, data_path):
    # Initiate policy
    linucb_policy_object = linucb_policy(K_arms = K_arms, d = d, alpha = alpha)
    
    # Instantiate trackers
    aligned_time_steps = 0
    cumulative_rewards = 0
    aligned_ctr = []
    unaligned_ctr = [] # for unaligned time steps
    
    # Open data
    with open(data_path, "r") as f:
        for line_data in f:
            if line_data == '\n':
                continue
            ################################################################################################
            # 1st column: Logged data arm. 
            # Integer data type
            data_arm = int(line_data.split()[0])

            # 2nd column: Logged data reward for logged chosen arm
            data_reward = float(line_data.split()[1])

            # 3rd columns onwards: 100 covariates. Keep in array of dimensions (100,) with float data type
            data_x_array = np.array([float(covariate_elem) for covariate_elem in line_data.split()[2:]])
            ################################################################################################

            ################################################################################################
            # Find policy's chosen arm based on input covariates at current time step
            # [ucb] select action (arm_index) based on given context (data_x_array)
            arm_index = linucb_policy_object.select_arm(data_x_array)

            # Check if arm_index is the same as data_arm (ie same actions were chosen)
            # Note that data_arms index range from 1 to 10 while policy arms index range from 0 to 9.
            if arm_index + 1 == data_arm:
                # Use reward information for the chosen arm to update
                # [ucb] update policy with reward given by environment after agent took action
                linucb_policy_object.linucb_arms[arm_index].reward_update(data_reward, data_x_array)

                # For CTR calculation
                aligned_time_steps += 1
                cumulative_rewards += data_reward
                aligned_ctr.append(cumulative_rewards/aligned_time_steps)
            ################################################################################################                
                    
    return (aligned_time_steps, cumulative_rewards, aligned_ctr, linucb_policy_object)


if __name__ =='__main__':
    alpha_input = 1.5
    data_path = "data.txt"
    aligned_time_steps, cum_rewards, aligned_ctr, policy = ctr_simulator(K_arms = 10, d = 100, alpha = alpha_input, data_path = data_path)
    print(aligned_time_steps, cum_rewards, aligned_ctr, policy)