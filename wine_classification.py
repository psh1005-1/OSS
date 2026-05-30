from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. 데이터셋 로드
wine = load_wine()
X = wine.data
y = wine.target

# 2. train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. 머신러닝 모델 학습
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. 예측
y_pred = model.predict(X_test)

# 5. 정확도 출력
accuracy = accuracy_score(y_test, y_pred)

print("Wine Classification Result")
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred, target_names=wine.target_names))

print("Train data size:", len(X_train))
print("Test data size:", len(X_test))