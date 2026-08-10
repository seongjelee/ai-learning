import numpy as np

dog = np.array([1.0, 2.0])
puppy = np.array([2.0, 3.0])
car = np.array([-2.0, 1.0])
X = np.array([
    dog,
    puppy,
    car
])

print(X.shape)

X_T = np.transpose(X)

print(X_T.shape)
S = X @ X_T

print(S)

dog_norm = np.linalg.norm(dog)
puppy_norm = np.linalg.norm(puppy)
car_norm = np.linalg.norm(car)

dog_normalized = dog / dog_norm
puppy_normalized = puppy / puppy_norm
car_normalized = car / car_norm

X_norm = np.array([
    dog_normalized,
    puppy_normalized,
    car_normalized
])

X_norm_T = np.transpose(X_norm)

S_norm = X_norm @ X_norm_T

print(S_norm)