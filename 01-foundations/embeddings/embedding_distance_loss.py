import numpy as np
import math

dog = np.array([1.0, 3.0])
puppy = np.array([4.0, 3.0])


learning_rate = 0.1
for i in range(100):
    distance = math.sqrt((dog[0]-puppy[0])**2 + (dog[1] - puppy[1])**2)
    Loss = (distance-1)**2

    gradient_x = 2 * (distance - 1) * ((dog[0]-4)/distance)

    dog[0] = dog[0] - learning_rate * gradient_x

    print(dog[0])
    print(distance)
    print(Loss)


#정확히 0이 되지 않는다. 학습률에 따라 기울기가 0이 되는 순간을 넘어서 더 큰 값을 학습할 수도 있기 때문이다
#0.01로 바꾸면 한 번 학습할 때 기울기와 loss도 조금씩밖에 조정되지 않지만, 그만큼 넘어가는 가능성도 줄어든다.
