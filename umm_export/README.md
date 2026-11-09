# umm (WM-811K 웨이퍼 불량 분류) 실험 옮기기

ver3 ~ ver7.X2 실험의 코드, 기록, 모델을 다른 컴퓨터에서 그대로 이어 쓰기 위한 묶음입니다. (2026-11-09 기준)

## 들어 있는 것

**umm_records.zip** (기록)
- `umm/` : 코드 (`wafixmatch.py`, `wafer/`, 평가 스크립트), `results/` (표·혼동행렬·마스킹 이미지), `runs/` (모든 실험의 로그 `log_*.txt`, `metrics.json`)
- `wafer_cache/` : 라벨 배열, 웨이퍼 크기(die 수) 배열, 불량 비율, 자기지도 학습용 10만 장 subset 인덱스 (seed0/seed1, ResNet-18/34)
- `notes/claude_memory/` : Claude 가 남긴 실험 메모. `wafixmatch-reproduction.md` 에 모든 버전의 설정과 결과가 시간순으로 있음
- `notes/analysis_scripts/` : 추가 분석 스크립트 (frozen probe, 크기별 혼동행렬, 치팅 검사, 결과표 이미지 등)
- `notes/env.txt` : Python / 패키지 버전 (pip freeze), GPU 정보

**umm_models.zip** (모델, 약 2.5GB)
- `wafer_cache/ver3/runs/.../*.pt` : 사전학습 인코더 (있으면 사전학습을 건너뜀)
- `umm/runs/.../best.pt` : ver3·ver6·ver7 계열 fine-tune 최고 모델

**빠진 것** (용량이 커서, 다시 만들 수 있음)
- `LSWMD.pkl` (원본 데이터 2GB) : Kaggle "WM-811K wafer map" 에서 받거나 기존 PC 의 `umm/LSWMD.pkl` 을 복사
- `wafer_cache/wm811k_96_packed.npy` (1.8GB) : 아래 `prepare` 로 다시 생성
- `wafer_cache/saliency/` : WaFixMatch 재현용. ver3~ver7 에는 필요 없음

## 설치 순서

1. 두 zip 을 같은 곳에 풀기. 결과 폴더:
   - `umm/` → 작업 폴더로 사용 (위치는 자유)
   - `wafer_cache/` → **홈 폴더 바로 아래로 이동** (`C:\Users\<사용자>\wafer_cache`). 코드가 `~/wafer_cache` 를 기본 경로로 읽음
2. `LSWMD.pkl` 을 `umm/` 안에 넣기
3. Python 3.11, `pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121`, 그다음 `numpy==2.4.6 scikit-learn==1.9.0` 등 `notes/env.txt` 참고
4. `umm/` 에서 `python wafixmatch.py prepare` (96x96 캐시 생성, 한 번만)
5. 확인: `python wafixmatch.py summary`

## 실행

```
python wafixmatch.py run --frac 0.05 --method ver7.X --seed 0
```

- 결과: `runs/resnet18/p0.05/seed<N>/<method>_ssl50_ft30/metrics.json`
- 이미 `done` 파일이 있는 폴더는 건너뜀 → 다시 돌리려면 그 폴더를 지우기
- 인코더 `.pt` 가 캐시에 있으면 사전학습을 건너뛰고 fine-tune 만 함 → 사전학습부터 다시 하려면 해당 `masklat*.pt` 를 지우기
- 사전학습은 에폭마다 `.ckpt` 를 저장하고 중간에 죽으면 다시 실행 시 이어서 진행 (끝나면 자동 삭제)
- `--seed` 는 자기지도 subset·사전학습·fine-tune 난수를 바꿈. 라벨 5% 와 val/test 분할은 `--split_seed` (기본 0) 로 고정

## 주의

- 시간: ResNet-18 사전학습 50에폭 ≈ 3.5~7시간 (RTX 3050 4GB 기준), fine-tune ≈ 15분
- GPU 작업은 한 번에 하나만 (RAM 16GB 에서 두 개 돌리면 멈춤)
- 노트북이면 전원 모드 "최고 성능", 충전기 연결, 전원 연결 시 절전 "안 함"
- 같은 seed 라도 GPU / 드라이버가 다르면 결과가 조금 달라질 수 있음. seed 0 → 1 만 바꿔도 ver7 macro 가 0.831 → 0.783 으로 흔들렸으므로 0.01~0.02 차이는 판단 근거로 약함
- 분석 스크립트는 경로를 인자로 받거나 홈 기준 경로를 씀. 실행 전 맨 위 경로 부분 확인
