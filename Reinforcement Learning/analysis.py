"""
Analysis question.
Change these default values to obtain the specified policies through value iteration.
If any question is not possible, return just the constant NOT_POSSIBLE:
```
return NOT_POSSIBLE
```
"""

NOT_POSSIBLE = None

def question2():
    """
    I found that with a noise of at most 0.01, the agent should be able to pass the bridge
    """

    answerDiscount = 0.9
    answerNoise = 0.01

    return answerDiscount, answerNoise

def question3a():
    """
    For taking the risky path to the +1 exit, I combined my process from questions 3b and 3c.
    By having a -1 living reward, the agent wants to take the shortest path to an exit. Having
    a lower discount rate will also prioritize the shorter term +1 exit over the greater +10
    exit.
    """

    answerDiscount = 0.75
    answerNoise = 0.2
    answerLivingReward = -1.0

    return answerDiscount, answerNoise, answerLivingReward

def question3b():
    """
    In order to take the safe route but end at the +1 exit, I found that lowering the discount
    rate to 0.3 would prioritize the shorter term succuess values.
    """

    answerDiscount = 0.3
    answerNoise = 0.2
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward

def question3c():
    """
    To prefer the distant +10 exit but take the risky path, adding a living reward of -1 was
    sufficient. It heavily penalizes the agent for taking longer paths so it takes the shorter
    one.
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = -1.0

    return answerDiscount, answerNoise, answerLivingReward

def question3d():
    """
    By default, the given parameters below already pass this test. Nothing was changed.
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward

def question3e():
    """
    By default, the given parameters below already pass this test. Nothing was changed.
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward

def question6():
    """
    No matter how much (or how little) randomness, the chance that the agent crosses the bridge
    completely is really low and would take more than 50 episodes.
    """

    return NOT_POSSIBLE

if __name__ == '__main__':
    questions = [
        question2,
        question3a,
        question3b,
        question3c,
        question3d,
        question3e,
        question6,
    ]

    print('Answers to analysis questions:')
    for question in questions:
        response = question()
        print('    Question %-10s:\t%s' % (question.__name__, str(response)))
