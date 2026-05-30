# OSS 개발 개인과제

동아대학교 OSS 개발 과목 개인과제 제출용 저장소입니다.

## 프로젝트 소개
Scikit-learn의 Wine Dataset을 활용하여 Random Forest 분류 모델을 구현하고, GitHub를 이용한 버전 관리 과정을 실습하였습니다.

## 사용 기술
- Python
- Scikit-learn
- GitHub

## 데이터셋
- Wine Dataset (`sklearn.datasets.load_wine`)

## 모델
- Random Forest Classifier

## 주요 기능
- 데이터셋 로드
- Train/Test 데이터 분할
- 모델 학습
- 예측 수행
- 정확도 평가
- n_estimators 파라미터 비교 실험

## 개발 및 Commit 내역

### 1차 Commit - 기본 모델 구현
- Wine Dataset 로드
- Train/Test 데이터 분할
- Random Forest 모델 학습
- 예측 및 정확도 출력

### 2차 Commit - 파라미터 변경 실험
- n_estimators 값 변경 실험
  - 10
  - 50
  - 100
  - 200
- 성능 비교 수행
- 최적 파라미터(n_estimators=100) 적용
- 최종 결과 출력

## 실행 방법

```bash
python wine_classification.py
