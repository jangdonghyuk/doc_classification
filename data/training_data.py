
# label: 0 - 지원 기간
period_data = [
    # 기본적인 날짜 표현
    {"text": "신청기간 : 2024. 1. 15.(월) ~ 2024. 2. 28.(수)", "label": 0},
    {"text": "접수기간 : '24. 3. 1.(금) ~ 3. 31.(일) 18:00", "label": 0},
    {"text": "모집기간 : 2024-04-01 ~ 2024-04-30", "label": 0},
    {"text": "지원신청 : 24.5.1 ~ 24.5.31", "label": 0},
    
    # 연도 표현 변형
    {"text": "신청기간 : 2024년 6월 1일 ~ 6월 30일", "label": 0},
    {"text": "접수기간 : '24년 7월 1일부터 7월 31일까지", "label": 0},
    {"text": "모집기간 : 2024.8.1.~2024.8.31.", "label": 0},
    
    # 시간 표현 포함
    {"text": "신청기간 : 2024.9.1.(일) 09:00 ~ 9.30.(월) 17:00", "label": 0},
    {"text": "접수마감 : 2024년 10월 31일 16:00까지", "label": 0},
    {"text": "□ 신청기간 : '24.11.1 10:00 ~ 11.30 18:00", "label": 0},
    
    # 특수문자 및 기호 변형
    {"text": "가. 신청기간: 2024-01-01~2024-01-31", "label": 0},
    {"text": "② 접수기간 →  '24.02.01.~'24.02.29.", "label": 0},
    {"text": "■ 모집기간 ▶ 2024.03.01.~03.31.", "label": 0},
    
    # 상시모집 표현
    {"text": "신청기간 : 상시모집 (예산 소진시까지)", "label": 0},
    {"text": "접수기간 : 상시접수 (정원 충족시 조기마감)", "label": 0},
    {"text": "모집기간 : 연중 상시", "label": 0},
    
    # 분기/반기 표현
    {"text": "신청기간 : 2024년 1분기 (1월~3월)", "label": 0},
    {"text": "접수기간 : '24년 상반기", "label": 0},
    {"text": "모집기간 : 2024년도 하반기(7월~12월)", "label": 0},
    
    # 공고일 기준 표현
    {"text": "신청기간 : 공고일로부터 30일 이내", "label": 0},
    {"text": "접수기간 : 공고일로부터 2024년 말까지", "label": 0},
    {"text": "모집기간 : 공고일로부터 ~ 예산 소진시", "label": 0},
    
    # 단계별 접수
    {"text": "1차 접수 : 2024.01.01.~01.31. / 2차 접수 : 2024.07.01.~07.31.", "label": 0},
    {"text": "상반기 접수('24.1.1~6.30), 하반기 접수('24.7.1~12.31)", "label": 0},
    {"text": "1차 모집 '24.3월, 2차 모집 '24.9월 예정", "label": 0},
    
    # 기타 다양한 표현
    {"text": "<접수기간> 2024년도 예산 소진시까지 상시접수", "label": 0},
    {"text": "- 신청기한 : ~ 2024.12.31. 18:00까지", "label": 0},
    {"text": "※ 접수기간 : 2024년 매월 1일~10일(18:00)까지", "label": 0},
    {"text": "□ 신청서 접수 : 격월 1~10일", "label": 0},
    
    # 특수한 경우
    {"text": "사업기간 종료시까지 수시 접수", "label": 0},
    {"text": "※ 신청・접수기간 : 수시", "label": 0},
    {"text": "- 모집기간 : 연간 수시모집", "label": 0},

    {"text": "11. 신청기간 : '24. 10. 30.(수) ~ 11. 8.(금) 18:00 까지", "label": 0},
    {"text": "신청기간 : '24. 3. 1.(금) ~ 3. 31.(일) 18:00", "label": 0},
    {"text": "지원신청 : 24.5.1 ~ 24.5.31", "label": 0},
    {"text": "□ 신청기간 : '24.11.1 10:00 ~ 11.30 18:00", "label": 0},
    {"text": "접수기간 : 2024년 매월 1일~10일(18:00)까지", "label": 0},
]

# label: 1 - 지원 내용
support_data = [
    # 기본 금액/비율 표현
    {"text": "지원금액: 총 사업비의 60%, 최대 1억원 한도로 지원합니다.", "label": 1},
    {"text": "사업비 지원: 기업당 최대 8천만원(총 사업비의 75% 이내)", "label": 1},
    
    # 복합적 지원 내용
    {"text": "1단계 지원 500만원, 2단계 지원 2,000만원 (기업당)", "label": 1},
    {"text": "기초진단 100만원, 심화컨설팅 400만원 범위 내 지원", "label": 1},
    
    # 정부/지자체 매칭
    {"text": "총 사업비 1억원(국비 40%, 도비 30%, 시비 30%)", "label": 1},
    {"text": "보조금 분담비율: 도비 2억원(40%), 시군비 3억원(60%)", "label": 1},
    
    # 상세 조건이 있는 경우
    {"text": "- 기업당 최대 20백만원, 상시근로자 수 대비 차등 지원\n- 50인 미만: 10백만원\n- 50~100인 미만: 15백만원\n- 100인 이상: 20백만원", "label": 1},
    {"text": "지원한도: 기업당 총 사업비의 50%, 최대 2.5억원 지원 (단, 100개사 내외)", "label": 1},
    
    # 복수 지원사항
    {"text": "- 근로자 복지지원금: 기업당 최대 20백만원\n- 육아휴직대체인턴: 월 50만원, 12개월\n- 추가고용장려금: 월 50만원, 6개월", "label": 1},
    
    # 지원제외/자부담 명시
    {"text": "지원규모: 35,200천원 (자부담 20% 이상, 도비 30%, 군비 70%)", "label": 1},

    {"text": "2. 총사업비 : 2,024백만원(국 1,124백만원, 시 900백만원) ※ 자부담 10% 별도", "label": 1},
    {"text": "지원금액: 총 사업비의 60%, 최대 1억원 한도로 지원", "label": 1},
    {"text": "사업비 지원: 기업당 최대 8천만원(총 사업비의 75% 이내)", "label": 1},
    {"text": "총 사업비 1억원(국비 40%, 도비 30%, 시비 30%)", "label": 1},
    {"text": "지원규모: 35,200천원 (자부담 20% 이상)", "label": 1},
]

# label: 2 - 기타
etc_data = [
    {"text": "자세한 내용은 공고문을 참고하시기 바랍니다.", "label": 2},
    {"text": "문의사항은 담당자에게 연락주시기 바랍니다.", "label": 2},
    {"text": "제출된 서류는 반환하지 않습니다.", "label": 2},
    {"text": "세부사항은 공고문을 참조하시기 바랍니다.", "label": 2}
]