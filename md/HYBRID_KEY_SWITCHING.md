# MOAI (Phantom-FHE) Hybrid Key Switching

MOAI의 CKKS 연산은 `thirdparty/phantom-fhe`를 통해 수행되며, 키스위칭(relinearize, rotation 등)은 **Microsoft SEAL 스타일의 hybrid key switching**을 Phantom 구현이 그대로 따른다. MOAI 전용으로 별도 수식을 다시 구현한 레이어는 없고, `EncryptionParameters`·`PhantomContext`·`DRNSTool`·`eval_key_switch.cu`·`secretkey.cu` 경로가 핵심이다.

## 1. 용어: 이 코드베이스에서의 `alpha`와 `dnum`

Phantom relin 키 생성 (`secretkey.cu`의 `generate_one_kswitch_key`) 기준:

| 이름 | 정의 |
|------|------|
| `size_P` | `parms.special_modulus_size()` — **특수 모듈러스에 쓰는 소수의 개수** (비트 수 아님) |
| `size_QP` | 키 레벨에서의 `coeff_modulus` 소수 개수 |
| `size_Q` | `size_QP - size_P` — 데이터에 해당하는 Q 쪽 소수 개수 |
| **`alpha`** | **`alpha = size_P`** (코드에서 동일 기호) |
| **`dnum`** | **`dnum = size_Q / size_P`** (정수로 나누어떨어져야 함) |

기본값은 `EncryptionParameters`에서 **`special_modulus_size_ = 1`** 이므로, 별도로 `set_special_modulus_size`를 호출하지 않으면:

- **`alpha = 1`** (특수 프라임 1개, 보통 체인 **맨 끝** 소수들이 P)
- **`dnum = size_Q`** = **전체 coeff_modulus 소수 개수 − 1**

예: coeff_modulus 소수가 16개이면 `size_Q = 15`, `dnum = 15`, `alpha = 1`.

## 2. 컨텍스트에서 P와 Q의 역할

`context.cu`에서:

- `size_P != 0`이면 `using_keyswitching_ = true`
- 데이터 레벨의 `coeff_modulus`에서는 **뒤에서부터 `size_P`개**를 제거한다 (특수 모듈러스는 데이터 체인에 남기지 않고 키스위칭용으로만 사용하는 구조).

즉 coeff_modulus 벡터는 **`[ … Q 쪽 소수들 … , … P 쪽 소수들 … ]`** 형태로 두고, 마지막 `size_P`개가 hybrid KS용 P다.

## 3. RNS 쪽에서 “hybrid”라고 부르는 부분

`rns.cu`의 `DRNSTool` 초기화에서 `size_P != 0`이면 주석 **`// hybrid key-switching`** 아래로 `bigP`, `bigP_mod_q` 등을 만들고, Ql을 `alpha` 크기로 쪼개 `beta = ceil(size_Ql / alpha)` 등으로 **mod-up / base conversion** 테이블을 쌓는다. 키스위칭 본연산은 `eval_key_switch.cu`의 `keyswitch_inplace` → `modup` → `key_switch_inner_prod` → `moddown` 흐름이다.

## 4. 다른 파일의 `alpha`와 혼동하지 말 것

- `evaluate.cu`의 `FindLevelsToDrop` 등에 나오는 **`alpha`는 노이즈 분석용 실수**이며, 위 **`size_P`와 같은 의미가 아니다.**

## 5. MOAI 예시 파라미터 (참고)

`test_ct_pt_matrix_mul.cuh` 등에서:

```cpp
parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree,
    {60, 40, 40, ..., 40, 60}));  // 총 16개 소수
```

기본 `special_modulus_size = 1` 가정 시:

- **`alpha = 1`**
- **`dnum = 15`**

실제 실험/배포에서 `set_special_modulus_size(k)`를 쓰면 `alpha = k`, `dnum = size_Q / k`로 바뀌므로, **항상 해당 빌드의 `EncryptionParameters` 설정을 확인**해야 한다.

## 6. 참고 파일

| 경로 | 내용 |
|------|------|
| `thirdparty/phantom-fhe/include/util/encryptionparams.h` | `special_modulus_size`, 기본값 1 |
| `thirdparty/phantom-fhe/src/context.cu` | P 제거, `using_keyswitching_` |
| `thirdparty/phantom-fhe/src/rns.cu` | hybrid KS RNS 도구 |
| `thirdparty/phantom-fhe/src/secretkey.cu` | `dnum`, `alpha`, relin 키 생성 |
| `thirdparty/phantom-fhe/src/eval_key_switch.cu` | `keyswitch_inplace` |
