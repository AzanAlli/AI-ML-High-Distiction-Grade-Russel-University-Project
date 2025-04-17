from deepface import DeepFace
demography = DeepFace.analyze("1.jpg", actions = ['age', 'gender', 'race', 'emotion'])
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])
