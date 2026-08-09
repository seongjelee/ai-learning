import numpy as np

dog = np.array([1.0, 2.0])
puppy = np.array([2.0, 3.0])
car = np.array([-2.0, 1.0])

d_p = dog[0]*puppy[0] + dog[1]*puppy[1]
d_c = dog[0]*car[0] + dog[1]*car[1]

dog_norm = np.sqrt(dog[0]**2 + dog[1]**2)
puppy_norm**2 = np.sqrt(puppy[0]**2 + puppy[1]**2)
car_norm**2 = np.sqrt(car[0]**2 + car[1]**2)

cosd_p = d_p/(dog_norm*puppy_norm)
cosd_c = d_c/(dog_norm*car_norm)


