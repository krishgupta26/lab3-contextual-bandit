Lab 3: Contextual Bandit-Based News Article Recommendation System

Course: Reinforcement Learning Fundamentals
Student Name: Krish Gupta
Roll Number: U202300058
Branch: krishgupta_U20230058

1. Project Overview

This project implements a Contextual Multi-Armed Bandit (CMAB) based News Recommendation System.

Unlike a standard Multi-Armed Bandit (MAB), which selects actions without contextual information, a Contextual Bandit observes side information (context) before choosing an arm. In this system:

Contexts: User categories (User1, User2, User3)

Arms: News categories (ENTERTAINMENT, EDUCATION, TECH, CRIME)

Total Arms: 12 (4 per context)

The objective is to learn a policy that selects the optimal news category conditioned on user type in order to maximize expected user engagement.

2. System Architecture

The system consists of four major components:

2.1 Data Preprocessing

Loaded the datasets: train_users.csv, test_users.csv, and news_articles.csv

Dropped irrelevant columns such as user_id

Handled missing values using SimpleImputer

Applied StandardScaler to numeric features

Applied OneHotEncoder to categorical features

Used a ColumnTransformer pipeline to ensure structured preprocessing

2.2 User Classification (Context Detection)

A HistGradientBoostingClassifier was trained on train_users.csv to predict:

User1

User2

User3

This classifier acts as the context detector for the contextual bandit system.

The model was evaluated on test_users.csv, and classification accuracy was reported in the notebook.

3. Contextual Bandit Algorithms

Three contextual bandit strategies were implemented:

Epsilon-Greedy

Upper Confidence Bound (UCB)

Softmax (τ = 1)

Each algorithm:

Trains a separate model for each user context

Runs for a time horizon of T = 10,000

Uses the provided rlcmab-sampler package to obtain rewards

4. Experimental Results

Final Average Reward after 10,000 steps:

User 1

Epsilon: 10.15

UCB: 11.18

Softmax: 11.19

User 2

Epsilon: 1.94

UCB: 2.18

Softmax: 1.72

User 3

Epsilon: 6.11

UCB: 6.87

Softmax: 6.66

5. Analysis and Observations

UCB achieved the highest average reward in two out of three contexts.

Softmax slightly outperformed UCB in User1 but was less consistent overall.

Epsilon-Greedy performance depended heavily on the choice of ε.

UCB demonstrated faster convergence and more stable long-term performance.

Proper exploration-exploitation balance significantly affects reward convergence.

Based on empirical performance and consistency, UCB (C = 1) was selected as the final policy for the recommendation engine.

6. Hyperparameter Sensitivity
Epsilon-Greedy

Low ε results in faster exploitation but risks suboptimal convergence.

High ε increases exploration but reduces long-term reward.

ε = 0.1 provided the best trade-off.

UCB

Low C reduces exploration.

High C slows convergence.

C = 1 produced the best overall performance.

Softmax

τ = 1 provided smooth probabilistic exploration.

Stable but slightly weaker than UCB overall.

7. Final Recommendation Engine

The final system integrates classification and contextual bandit decision-making:

The classifier predicts the user category.

The trained UCB model selects the best news category.

An article is randomly sampled from the selected category.

The article is returned as the final recommendation.

This creates a complete end-to-end contextual reinforcement learning recommendation system.

8. Plots Included

The notebook includes:

Average Reward vs Time for each context

Hyperparameter comparison plots for ε

Hyperparameter comparison plots for C

Final algorithm comparison plot

All plots include labeled axes, legends, and descriptive titles as required.

9. Reproducibility Instructions
Requirements

Python 3.12

numpy

pandas

matplotlib

scikit-learn

rlcmab-sampler

Installation
pip install numpy pandas matplotlib scikit-learn rlcmab-sampler

Running the Project

Open the notebook:
lab3_results_U202300058.ipynb

Ensure the correct roll number is used:

reward_sampler = sampler(58)


Run all cells sequentially.

10. Conclusion

This project demonstrates how contextual reinforcement learning can be applied to personalized recommendation systems.

Among the evaluated strategies, UCB achieved the most consistent and highest average reward across contexts, making it the most suitable algorithm for deployment in this recommendation framework.
