## On-the-fly eval-key `a` 생성 & eval-key(리니어라이즈/로테이트 키) 생성 원리 정리

이 문서는 MOAI/Phantom 기반 CKKS 구현에서

- **eval-key(=relin key, galois/rotate key)** 가 어떤 수식으로 만들어지는지
- 그 과정에서 **(b,a)** 중 **`a`가 왜/어떻게 on-the-fly 생성 가능한지**
- Phantom 코드가 실제로 **`a`를 coeff-domain에서 만든 뒤 NTT하는지**, 아니면 **NTT-domain 값으로 바로 취급하는지**

를 한 번에 정리한다.

---

## 1) RLWE 암호문(또는 encrypt-zero)의 표준 형태

대부분의 RLWE 기반 스킴(CKKS/BFV/BGV)의 2-성분 암호문은 대략

\[
  (c_0, c_1) \equiv (b, a)
\]

로 볼 수 있고, secret key \(s\) 하에서 다음 관계를 만족하도록 만든다(부호 컨벤션은 구현마다 ±가 조금 다를 수 있음):

- \(a \leftarrow U(R_q)\) (uniform random poly)
- \(e \leftarrow \chi\) (noise poly)
- 평문(또는 타깃) \(m\)

\[
  b = -a\cdot s + e + m \pmod q
\]

특히 **0을 암호화(encrypt-zero)** 하면 \(m=0\)이므로

\[
  (b_0, a) = (-a\cdot s + e,\ a)
\]

형태의 “0에 대한 RLWE 샘플”이 된다.

---

## 2) eval-key가 “0 암호화로 시작해도 되는” 핵심 원리

eval-key(=키스위칭 키)는 본질적으로 어떤 타깃 다항식 \(m\) (예: \(s^2\), 또는 회전된 \(s^{(g)}\))를
현재 secret key \(s\) 하에서 암호화한 RLWE 샘플을 여러 개(digit/tower) 들고 있는 것이다.

위 식을 보면

\[
  (-a s + e + m,\ a) = (-a s + e,\ a) + (m, 0)
\]

즉,

- 먼저 **encrypt-zero**로 \((b_0,a)=(-as+e, a)\) 를 만들고
- 그 다음 **첫 성분에만** \(m\) 을 더해 \(b=b_0+m\) 으로 만들면

그게 곧바로 **\(m\)을 담은 암호문**이 된다.

그래서 eval-key 생성은 흔히 다음처럼 구현된다:

1) `encrypt_zero_symmetric(...)` 로 \((-(as+e), a)\) 생성  
2) 어떤 커널로 \(P_{w,q}(m)\) 같은 타깃 항을 **c0에 더함**  

Phantom의 주석도 이 형태를 직접 말한다:

- `Every pk_ = [P_{w,q}(s^2)+(-(as+e)), a]`

---

## 3) Phantom에서 eval-key 생성은 public-key 암호화를 쓰나?

Phantom/MOAI 기준:

- 일반 plaintext 암호화(입력 데이터를 ct로 만드는)는 **public key(비대칭) 경로**를 사용 가능
- 하지만 **eval-key(relin/galois) 생성은 secret key 기반(symmetric) encrypt-zero**를 사용한다.

이는 `PhantomSecretKey::generate_one_kswitch_key(...)` 내부에서
`encrypt_zero_symmetric(..., is_ntt_form=true, ...)` 를 호출해 tower를 만든 다음,
타깃 항을 c0에 더하는 흐름이기 때문이다.

---

## 4) Phantom에서 eval-key의 `a`는 coeff-domain에서 만들고 NTT를 적용하나?

Phantom의 `encrypt_zero_symmetric(..., is_ntt_form=true)` 경로를 보면:

- noise \(e\)는 coeff-domain에서 샘플링한 뒤 **NTT를 적용**한다.
- 반면 `a`는 `sample_uniform_poly(c1, prng_seed_a, ...)`로 **`c1=a`를 채운 뒤**
  별도의 `NTT(a)` 호출 없이 곧바로 NTT-domain 곱(예: `multiply_and_add_negate_rns_poly`)에 들어간다.

즉 코드 관점에서 **`a`에 대해 “샘플 → NTT” 단계가 존재하지 않는다.**
따라서 Phantom 구현은 사실상 “`a`를 NTT-domain에서 사용할 값으로 바로 공급”하는 형태다
(단, `sample_uniform_poly`가 내부적으로 어떤 통계를 목표로 하는지는 별개로, 적어도 별도 NTT 변환 커널 호출은 없다).

---

## 5) `a`는 limb(RNS modulus) 단위로 생성되는가?

네. `sample_uniform_poly(out, prng_seed, modulus[], poly_degree, coeff_mod_size)`는
출력 버퍼가 길이 \(N \times \text{coeff\_mod\_size}\)이고,
커널이 각 인덱스를 **어느 tower/limb \(q_j\)** 인지 구분해서
그 modulus \(q_j\)에 대해 계수를 생성한다.

즉 `a`는 RNS 표현의 각 limb \(a_j\)가 존재하며,
각 limb마다 \(0 \le a_{j,i} < q_j\) 균일 샘플(목표)을 생성한다.

---

## 6) 왜 (b,a) 중 `a`만 on-the-fly 생성하는 게 가능/유용한가?

eval-key는 (b,a)로 저장되며, 여기서

- `a`는 **uniform random poly**라서 seed/metadata 기반으로 재생성 가능
- `b`는 `-(a*s + e)`에 타깃 항까지 더한 결과라서, 일반적으로 키에 저장되어야 함(또는 별도의 방법으로 압축/재구성 필요)

따라서 “키 트래픽을 줄이기 위해 `a`만 온칩에서 재생성”은 자연스러운 최적화다.

또한 NTT는 bijection이므로, “coeff-domain에서 `a`를 뽑고 NTT”와
“NTT-domain에서 바로 `a_hat`를 뽑아 사용”을 동일 분포로 맞출 수 있다면(분포를 uniform으로 유지),
수학적으로도 정합성 문제가 없다(보안/구현상의 주의는 별도).

---

## 7) Reject sampling(모듈러 바이어스 제거) 요약

각 limb modulus \(q\)에 대해 64-bit word \(x\)를 XOF에서 읽고

\[
T = \left\lfloor \frac{2^{64}}{q} \right\rfloor \cdot q
\]

- \(x < T\) 이면 accept, 출력 \(x \bmod q\)
- 아니면 reject 후 다음 word

이 방식은 \(0..q-1\)에 대한 modulo bias를 제거한다.

---

## 8) 용어 매핑(실수 방지)

- public key \((pk0, pk1)\)는 **키 생성 시 고정**되고, 암호화마다 바뀌는 건 \(u,e\) 등 랜덤.
- eval-key(리니어라이즈/로테이트 키)는 “0 암호화 + 타깃 항 덧셈”으로 만드는 것이 정상적인 구현 패턴이다.
- Phantom에서 `encrypt_zero_symmetric(..., is_ntt_form=true)`는 **NTT-domain 키/암호문 생성 경로**로 쓰이며,
  이때 `a`는 별도 NTT 변환 호출 없이 곧바로 NTT-domain 연산으로 들어간다.

