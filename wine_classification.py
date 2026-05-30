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

# 3. n_estimators 파라미터 변경 실험
for n in [10, 50, 100, 200]:
    model = RandomForestClassifier(n_estimators=n, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"n_estimators={n:>3} | Accuracy: {accuracy:.4f}")

# 4. 최적 파라미터(n_estimators=100)로 최종 결과 출력
print("\n[ 최종 모델 결과 (n_estimators=100) ]")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=wine.target_names))

print("Train data size:", len(X_train))
print("Test data size:", len(X_test))
