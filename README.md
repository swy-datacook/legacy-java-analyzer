# Legacy Java Analyzer

레거시 Java(Spring/MyBatis) 프로젝트를 분석해 AI 컨텍스트용 JSON을 생성하는 Python CLI 도구입니다.

## 설치

프로젝트 루트에서 editable 설치:

```bash
pip install -e .
```

설치 후 CLI:

```bash
legacy-analyzer --root /path/to/legacy-java-project --output /path/to/output/legacy-context.json --pretty
```

## 쉘 스크립트 실행(로컬 개발용)

```bash
./legacy-analyze.sh /path/to/legacy-java-project /path/to/output/legacy-context.json
```

## LSP 포함 실행(선택)

기본은 정적 분석만 수행합니다.

```bash
legacy-analyzer --root /path/to/legacy-java-project --output ./analysis/legacy-context-lsp.json --pretty --include-lsp
```

`jdtls` 인자를 직접 줄 경우:

```bash
legacy-analyzer --root /path/to/legacy-java-project --output ./analysis/legacy-context-lsp.json --pretty --include-lsp --lsp-command "jdtls -data .jdtls-workspace"
```

LSP 서버가 없거나 초기화 실패해도 전체 분석은 계속되며 `lsp.status`에 기록됩니다.

## 출력 JSON 주요 섹션

- `metadata`: 분석 시각/경로
- `project`: 파일 확장자 통계, 상위 구조
- `build.maven`: pom 의존성/플러그인/프로파일
- `java`: 패키지/레이어/의존 그래프(import 기반)
- `mybatis`: Mapper Java/XML 현황, sqlMap 잔존량
- `spring`: `web.xml` 및 context XML 요약
- `risks`: 설정 파일 내 잠재 credential 패턴 위치
- `lsp`: LSP 심볼 조회 결과(선택)
- `aiContext`: 후속 마이그레이션 입력용 요약 카드
