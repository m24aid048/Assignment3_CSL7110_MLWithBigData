# Recommender Systems: Content-Based and Collaborative Filtering
| **Name: -** | Shashi Saurav |
| **Roll No: -** | M24AID048 |
| **Course: -** | CSL7110 – Machine Learning with Big Data |
| **Institution: -** | IIT Jodhpur |
| **GitHub Link: -** | https://github.com/m24aid048/Assignment3_CSL7110_MLWithBigData.git|
## Overview

This project implements multiple recommender system techniques using the **MovieLens dataset**. The objective is to build, evaluate, and compare different recommendation approaches including **Content-Based Filtering, Collaborative Filtering, Matrix Factorization, Hybrid Models, Neural Network Recommenders, Reinforcement Learning approaches, and Explainability techniques**.

The implementation is provided in a **Jupyter Notebook** and executed in **Google Colab** using Python.

The goal is to recommend relevant movies to users based on historical ratings and movie metadata while also providing interpretable explanations for recommendations. 

---

# Dataset

The project uses the **MovieLens dataset**, which contains:

* **movies.csv**

  * Movie ID
  * Title
  * Genres

* **ratings.csv**

  * User ID
  * Movie ID
  * Rating (1–5)
  * Timestamp

The dataset is widely used as a benchmark for recommender system research and provides user-movie interaction data required for building recommendation models. 

---

# Implementation Structure

The notebook is divided into **six major parts** corresponding to the tasks in the assignment.

---

# Part 1: Content-Based Filtering

## Task 1: TF-IDF Based Recommendation

A content-based recommender system is implemented using **TF-IDF vectorization of movie genres**.

### Method

1. Extract movie genres.
2. Convert genres into **TF-IDF vectors**.
3. Compute **cosine similarity** between movie vectors.
4. Recommend the **Top-N most similar movies** for a given movie title.

### Output

* List of similar movies
* Cosine similarity score

---

## Task 2: User Profile Based Content Recommender

A **user profile vector** is constructed from the movies the user has rated.

### Method

The user profile is computed as a weighted average of movie feature vectors:

$$
P_u = \frac{\sum r_{u,m} f_m}{\sum r_{u,m}}
$$

Where:

* (r_{u,m}) = rating by user *u* for movie *m*
* (f_m) = feature vector of movie *m*

### Steps

1. Retrieve movies rated by a user.
2. Multiply TF-IDF vectors by user ratings.
3. Normalize the resulting vector.
4. Compute cosine similarity between user profile and movie vectors.

### Evaluation

* Precision@K
* Recall@K

---

# Part 2: Collaborative Filtering

## Task 3: User-Based Collaborative Filtering

User-based collaborative filtering recommends movies based on **similar users' preferences**.

### Method

1. Create a **user-movie rating matrix**.
2. Compute **user similarity** using cosine similarity.
3. Identify **K nearest neighbors**.
4. Predict ratings using weighted average of neighbor ratings.

### Evaluation Metrics

* RMSE
* Precision@K
* Recall@K

---

## Task 4: Item-Based Collaborative Filtering

Item-based collaborative filtering recommends movies based on **similarity between items**.

### Method

1. Compute **item-item similarity matrix**.
2. Identify movies rated by a user.
3. Use ratings of similar items to predict new ratings.
4. Recommend top-rated items.

### Advantage

Item-based CF is generally more scalable and memory efficient for large datasets.

---

# Part 3: Matrix Factorization

## Task 5: Singular Value Decomposition (SVD)

Matrix factorization decomposes the user-item matrix into latent factors.

$$
R \approx U \Sigma V^T
$$

Where:

* **U** = user latent factors
* **Σ** = singular values
* **Vᵀ** = item latent factors

### Steps

1. Construct rating matrix.
2. Apply **SVD decomposition**.
3. Reconstruct approximate rating matrix.
4. Generate Top-N recommendations.

---

## Task 6: Matrix Factorization Using Surprise Library

The **Surprise library** is used to train an SVD recommendation model.

### Steps

1. Load data into Surprise framework.
2. Split into training and testing sets.
3. Train SVD model.
4. Evaluate performance.

### Metrics

* RMSE
* Precision@K
* Recall@K

---

# Part 4: Hybrid Recommendation Model

## Task 7: Hybrid Recommender

A hybrid recommender combines **content-based filtering and collaborative filtering**.

$$
FinalScore = \alpha \times CBF + (1-\alpha) \times CF
$$

Where:

* CBF = Content-Based score
* CF = Collaborative Filtering score
* α = weighting parameter

### Steps

1. Compute CBF predictions.
2. Compute CF predictions.
3. Combine scores.
4. Train a meta-model using:

   * CBF score
   * CF score
   * Movie average rating
   * User average rating

### Evaluation

* RMSE
* Precision@K
* Recall@K

---

# Part 5: Learning-Based Recommender Systems

## Task 8: Neural Network Based Recommender

A neural network is used to learn **user and movie embeddings**.

### Inputs

* Movie features:

  * Genres
  * Release year
  * Average rating

* User features:

  * Average rating behaviour

### Architecture

```
User Input → Dense Layer → User Embedding
Movie Input → Dense Layer → Movie Embedding
Combine Embeddings → Dense Layers → Predicted Rating
```

### Training

* Loss Function: Mean Squared Error (MSE)
* Optimizer: Adam

The model learns complex user-movie relationships beyond traditional content-based filtering.

---

## Task 9: Reinforcement Learning Recommender

A reinforcement learning recommender treats recommendation as a **sequential decision problem**.

### RL Components

Agent: recommender system
State: user interaction history
Action: recommended movie
Reward: user feedback (rating)

### Implementation

A **Multi-Armed Bandit model with ε-greedy exploration** is implemented.

* With probability **ε = 0.1**, explore new movies.
* Otherwise recommend highest reward movie.

### Learning

Q-values are updated using a simplified Q-learning rule:

$$
Q(s,a) = Q(s,a) + \alpha(r - Q(s,a))
$$

---

# Part 6: Explainability

## Task 10: Feature-Based Explanations

Recommendations are explained using movie features.

Example explanation:

> “This movie was recommended because you like Action and Sci-Fi movies.”

Genre overlap between user preferences and movie features provides interpretable explanations.

---

## Task 11: Neighborhood-Based Explanations

Recommendations are explained using **similar users or items**.

Example:

> “Users who liked *Inception* also liked *Interstellar*.”

This uses **user similarity or item similarity**.

---

## Task 12: Model-Agnostic Explanations

The **LIME (Local Interpretable Model-Agnostic Explanations)** framework is used to explain predictions of the neural network model.

LIME identifies which features contribute most to the predicted rating.

---

## Task 13: Evaluating Explainability

Explainability methods are evaluated based on:

* Clarity of explanations
* Ability to justify recommendations
* Detection of potential bias in recommendation models

---

# Evaluation Metrics

The following metrics are used to evaluate recommendation quality:

### RMSE

Measures rating prediction error.

### Precision@K

Proportion of recommended items that are relevant.

### Recall@K

Proportion of relevant items that are recommended.

### Top-N Recommendation Evaluation

Measures how well the system ranks relevant items in top recommendations.

---

# Technologies Used

* Python
* Google Colab
* Pandas
* NumPy
* Scikit-Learn
* TensorFlow / Keras
* Surprise Library
* LIME

---

# Conclusion

This project demonstrates the implementation of multiple recommender system techniques and their comparison.

Key findings:

* Content-based filtering works well for cold-start items.
* Collaborative filtering captures user similarity effectively.
* Matrix factorization improves prediction accuracy.
* Hybrid models combine strengths of different approaches.
* Neural networks capture complex relationships between users and items.
* Reinforcement learning enables adaptive recommendation strategies.
* Explainability techniques improve transparency and trust in recommender systems.

---

# How to Run

1. Open the Jupyter Notebook in **Google Colab**.
2. Install required libraries.
3. MovieLens dataset uploaded in my GDrive and public Link provided in the code.
4. Run all cells sequentially.

---

# References

* MovieLens Dataset (GroupLens Research)
* Surprise Recommendation Library
* LIME Explainable AI Framework
* Recommender Systems Research Literature
